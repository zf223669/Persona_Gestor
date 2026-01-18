"""Training and evaluation"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torch
import tensorflow as tf
from models.utils import get_noise_fn
import torch.autograd.forward_ad as fwAD
from samplers.utils import NoiseScheduleVP
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import sde_lib
from models import ddpm, ncsnv2, ncsnpp
import losses
import numpy as np
from tqdm import tqdm
import time
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
flags.DEFINE_integer("n_batch", 1, "Number of batches per GPU", lower_bound=1)
flags.DEFINE_integer("batch_size", 512, "Batch size per GPU", lower_bound=1)
flags.DEFINE_integer("n_timesteps", 1200, "Number of timesteps", lower_bound=1)
flags.DEFINE_float("eps", 1e-3, "Eps")

tf.config.experimental.set_visible_devices([], "GPU")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def restore_checkpoint(state, loaded_state):
    state["optimizer"].load_state_dict(loaded_state["optimizer"])
    state["model"].load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]
    return state


def get_dataset_multi_host(config, batch_size, num_slices=8, slice=0):
    # Reduce this when image resolution is too large and data pointer is stored
    prefetch_size = tf.data.experimental.AUTOTUNE

    # Create dataset builders for each dataset.
    if config.data.dataset == "CIFAR10":
        dataset_builder = tfds.builder("cifar10")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

    # Customize preprocess functions for each dataset.

    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = resize_op(d["image"])

        return dict(image=img, label=d.get("label", None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.threading.private_threadpool_size = 48
        dataset_options.threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(split=split, shuffle_files=False, read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)
        ds = ds.shard(num_slices, slice)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(prefetch_size)
        return ds

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    print(f"Load dataset slice {slice}/{num_slices}, trainset length {len(train_ds)}, evalset length {len(eval_ds)}")
    return train_ds, eval_ds, dataset_builder


def get_time_steps(ns, skip_type, t_T, t_0, N, device):
    """Compute the intermediate time steps for sampling.

    Args:
        skip_type: A `str`. The type for the spacing of the time steps. We support three types:
            - 'logSNR': uniform logSNR for the time steps.
            - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
            - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
        t_T: A `float`. The starting time of the sampling (default is T).
        t_0: A `float`. The ending time of the sampling (default is epsilon).
        N: A `int`. The total number of the spacing of the time steps.
        device: A torch device.
    Returns:
        A pytorch tensor of the time steps, with the shape (N + 1,).
    """
    if skip_type == "logSNR":
        lambda_T = ns.marginal_lambda(torch.tensor(t_T).to(device))
        lambda_0 = ns.marginal_lambda(torch.tensor(t_0).to(device))
        logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
        return ns.inverse_lambda(logSNR_steps)
    elif skip_type == "time_uniform":
        return torch.linspace(t_T, t_0, N + 1).to(device)
    elif skip_type == "time_quadratic":
        t_order = 2
        t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
        return t
    else:
        raise ValueError(
            "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
        )


@torch.no_grad()
def get_noise_and_jvp_x(x, t, v, noise_pred_fn):
    def fn(data):
        return noise_pred_fn(data, torch.ones(data.shape[0], device=data.device) * t)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        noise, noise_jvp_x = fwAD.unpack_dual(fn(dual_x))
    return noise, noise_jvp_x


@torch.no_grad()
def get_noise_and_total_derivative(x, t, noise_pred_fn, ns):
    def fn(data, time):
        return noise_pred_fn(data, torch.ones(x.shape[0], device=x.device) * time)

    alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
    with fwAD.dual_level():
        vt = torch.ones_like(t)
        dual_t = fwAD.make_dual(t, vt)
        _, d_lambda_d_t = fwAD.unpack_dual(ns.marginal_lambda(dual_t))
        noise = fn(x, t)

        vt = torch.ones_like(t) / d_lambda_d_t
        vx = sigma_t**2 * x - sigma_t * noise
        dual_x = fwAD.make_dual(x, vx)
        dual_t = fwAD.make_dual(t, vt)
        _, noise_jvp = fwAD.unpack_dual(fn(dual_x, dual_t))
    return noise, noise_jvp


def compute_l(
    statistics_dir, MAX_BATCH, config, eps, n_timesteps, batch_size, num_gpus, ns, sde, loaded_state, device, r
):
    torch.cuda.set_device(r)
    config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scaler = datasets.get_data_scaler(config)
    train_ds, _, _ = get_dataset_multi_host(config, batch_size, num_slices=num_gpus, slice=r)
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(state, loaded_state)
    ema.copy_to(score_model.parameters())
    noise_pred_fn = get_noise_fn(sde, score_model, train=False, continuous=True)
    timesteps = get_time_steps(ns, "logSNR", sde.T, eps, n_timesteps, device)
    if os.path.exists(os.path.join(statistics_dir, f"l_{r}.npz")):
        return
    l_lst = [0] * len(timesteps)
    with torch.no_grad():
        for j, t in tqdm(enumerate(timesteps), desc="Computing l..."):
            time_start = time.time()
            for i, batch in enumerate(iter(train_ds)):
                if i >= MAX_BATCH:
                    break
                time_spent = time.time() - time_start
                print(f"Batch {i}/{MAX_BATCH}, {time_spent:.2f} s")
                train_batch = torch.from_numpy(batch["image"]._numpy()).to(device).float()
                train_batch = train_batch.permute(0, 3, 1, 2)
                x = scaler(train_batch)

                v = torch.randint(0, 2, x.shape, device=device) * 2.0 - 1
                z = torch.randn_like(x)
                alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
                perturbed_data = alpha_t * x + sigma_t * z
                _, noise_jvp_x = get_noise_and_jvp_x(perturbed_data, t, v, noise_pred_fn)
                l = (sigma_t * noise_jvp_x * v).mean(dim=0).cpu().numpy()
                l_lst[j] += l
            l_lst[j] = l_lst[j] / MAX_BATCH
    l_lst = np.asarray(l_lst)
    np.savez_compressed(os.path.join(statistics_dir, f"l_{r}.npz"), l=l_lst)


def collect_l(statistics_dir):
    print("Collecting l...")
    l_lsts = []
    for file in os.listdir(statistics_dir):
        if file.startswith("l_") and not file.startswith("l_d"):
            l_lst = np.load(os.path.join(statistics_dir, file))["l"]
            l_lsts.append(l_lst)
    np.savez_compressed(os.path.join(statistics_dir, "l.npz"), l=np.mean(l_lsts, axis=0))


def compute_l_d(statistics_dir, lambda_0, lambda_T):
    l_lst = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
    print("Computing l_d...")
    l_len = len(l_lst)
    l_d_lst = []
    gap = (lambda_0 - lambda_T) / (l_len - 1)
    for i in range(l_len):
        if i == 0:
            l_d_lst.append((l_lst[i + 1] - l_lst[i]) / gap)
        elif i == l_len - 1:
            l_d_lst.append((l_lst[i] - l_lst[i - 1]) / gap)
        else:
            l_d_lst.append((l_lst[i + 1] - l_lst[i - 1]) / (2 * gap))

    window = 5
    l_d_smooth_lst = []
    for i in range(l_len):
        if i < window:
            l_d_smooth_lst.append(np.sum(l_d_lst[: i + window + 1], axis=0) / (i + window + 1))
        elif i >= l_len - window:
            l_d_smooth_lst.append(np.sum(l_d_lst[i - window : l_len], axis=0) / (l_len - i + window))
        else:
            l_d_smooth_lst.append(np.sum(l_d_lst[i - window : i + window + 1], axis=0) / (2 * window + 1))
    l_d_smooth_lst = np.asarray(l_d_smooth_lst)
    np.savez_compressed(os.path.join(statistics_dir, "l_d.npz"), l_d=l_d_smooth_lst)


def compute_f(
    statistics_dir, MAX_BATCH, config, eps, n_timesteps, batch_size, num_gpus, ns, sde, loaded_state, device, r
):
    torch.cuda.set_device(r)
    config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scaler = datasets.get_data_scaler(config)
    train_ds, _, _ = get_dataset_multi_host(config, batch_size, num_slices=num_gpus, slice=r)
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(state, loaded_state)
    ema.copy_to(score_model.parameters())
    noise_pred_fn = get_noise_fn(sde, score_model, train=False, continuous=True)
    timesteps = get_time_steps(ns, "logSNR", sde.T, eps, n_timesteps, device)
    if os.path.exists(os.path.join(statistics_dir, f"f_{r}.npz")):
        return
    l_lst = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
    l_d_lst = np.load(os.path.join(statistics_dir, "l_d.npz"))["l_d"]
    f = [0] * len(timesteps)
    f_d = [0] * len(timesteps)
    f_f = [0] * len(timesteps)
    f_f_d = [0] * len(timesteps)

    with torch.no_grad():
        for j, t in tqdm(enumerate(timesteps), desc="Computing f..."):
            time_start = time.time()
            for i, batch in enumerate(iter(train_ds)):
                if i >= MAX_BATCH:
                    break
                time_spent = time.time() - time_start
                print(f"Batch {i}/{MAX_BATCH}, {time_spent:.2f} s")
                train_batch = torch.from_numpy(batch["image"]._numpy()).to(device).float()
                train_batch = train_batch.permute(0, 3, 1, 2)
                x = scaler(train_batch)

                z = torch.randn_like(x)
                alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
                perturbed_data = alpha_t * x + sigma_t * z
                noise, eps_d = get_noise_and_total_derivative(perturbed_data, t, noise_pred_fn, ns)
                l = torch.from_numpy(l_lst[j]).to(device)
                l_d = torch.from_numpy(l_d_lst[j]).to(device)
                lamb = ns.marginal_lambda(t)

                a = (sigma_t * noise - l * perturbed_data) / alpha_t
                b = torch.exp(-lamb) * ((l - 1) * noise + eps_d) - l_d * perturbed_data / alpha_t
                f[j] += a.mean(dim=0).cpu().numpy()
                f_d[j] += b.mean(dim=0).cpu().numpy()
                f_f[j] += (a * a).mean(dim=0).cpu().numpy()
                f_f_d[j] += (a * b).mean(dim=0).cpu().numpy()
            f[j] /= MAX_BATCH
            f_d[j] /= MAX_BATCH
            f_f[j] /= MAX_BATCH
            f_f_d[j] /= MAX_BATCH
    f = np.asarray(f)
    f_d = np.asarray(f_d)
    f_f = np.asarray(f_f)
    f_f_d = np.asarray(f_f_d)
    np.savez_compressed(os.path.join(statistics_dir, f"f_{r}.npz"), f=f, f_d=f_d, f_f=f_f, f_f_d=f_f_d)


def collect_f(statistics_dir):
    print("Collecting f...")
    f_lsts, f_d_lsts, f_f_lsts, f_f_d_lsts = [], [], [], []
    for file in os.listdir(statistics_dir):
        if file.startswith("f_"):
            fs_lsts = np.load(os.path.join(statistics_dir, file))
            f_lst, f_d_lst, f_f_lst, f_f_d_lst = fs_lsts["f"], fs_lsts["f_d"], fs_lsts["f_f"], fs_lsts["f_f_d"]
            f_lsts.append(f_lst)
            f_d_lsts.append(f_d_lst)
            f_f_lsts.append(f_f_lst)
            f_f_d_lsts.append(f_f_d_lst)
    np.savez_compressed(
        os.path.join(statistics_dir, "f.npz"),
        f=np.mean(f_lsts, axis=0),
        f_d=np.mean(f_d_lsts, axis=0),
        f_f=np.mean(f_f_lsts, axis=0),
        f_f_d=np.mean(f_f_d_lsts, axis=0),
    )


def compute_sb(statistics_dir):
    print("Computing s, b...")
    fs_lst = np.load(os.path.join(statistics_dir, "f.npz"))
    f, f_d, f_f, f_f_d = fs_lst["f"], fs_lst["f_d"], fs_lst["f_f"], fs_lst["f_f_d"]
    s = (f_f_d - f * f_d) / (f_f - f * f)
    b = f_d - s * f
    np.savez_compressed(os.path.join(statistics_dir, "sb.npz"), s=s, b=b)


def compute_lsb(opt):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """

    config = opt.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Create data normalizer and its inverse
    workdir = opt.workdir
    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unsupported.")

    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)
    timesteps = get_time_steps(ns, "logSNR", sde.T, opt.eps, opt.n_timesteps, "cpu")
    logSNR_steps = ns.marginal_lambda(timesteps)
    lambda_T = ns.marginal_lambda(torch.tensor(sde.T)).item()
    lambda_0 = ns.marginal_lambda(torch.tensor(opt.eps)).item()

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        if not tf.io.gfile.exists(ckpt_path):
            logging.warning(f"No checkpoint found at {ckpt_path}.")
            continue

        loaded_state = torch.load(ckpt_path, map_location="cpu")
        statistics_dir = os.path.join(
            workdir, "statistics", f"{ckpt}_{opt.eps}_{opt.n_timesteps}_{num_gpus}_{opt.n_batch}_{opt.batch_size}"
        )
        os.makedirs(statistics_dir, exist_ok=True)

        import torch.multiprocessing as mp

        mp.set_start_method(method="spawn", force=True)
        processes_l = [
            mp.Process(
                target=compute_l,
                args=(
                    statistics_dir,
                    opt.n_batch,
                    opt.config,
                    opt.eps,
                    opt.n_timesteps,
                    opt.batch_size,
                    num_gpus,
                    ns,
                    sde,
                    loaded_state,
                    device,
                    i,
                ),
            )
            for i in range(num_gpus)
        ]

        [p.start() for p in processes_l]
        [p.join() for p in processes_l]

        collect_l(statistics_dir)

        compute_l_d(statistics_dir, lambda_0, lambda_T)

        processes_f = [
            mp.Process(
                target=compute_f,
                args=(
                    statistics_dir,
                    opt.n_batch,
                    opt.config,
                    opt.eps,
                    opt.n_timesteps,
                    opt.batch_size,
                    num_gpus,
                    ns,
                    sde,
                    loaded_state,
                    device,
                    i,
                ),
            )
            for i in range(num_gpus)
        ]

        [p.start() for p in processes_f]
        [p.join() for p in processes_f]

        collect_f(statistics_dir)

        compute_sb(statistics_dir)


def main(argv):
    compute_lsb(FLAGS)


if __name__ == "__main__":
    app.run(main)
