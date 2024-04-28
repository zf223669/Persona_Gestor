import torch
from WavLM import WavLM, WavLMConfig
from os import path

# load the pre-trained checkpoints
print(path.exists("../path"))
checkpoint = torch.load('../path/to/WavLM-Large.pt')

cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the representation of last layer
wav_input_16khz = torch.randn(1, 10000)
print(wav_input_16khz)
if cfg.normalize:
    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
rep = model.extract_features(wav_input_16khz)[0]
print(rep.shape)
# extract the representation of each layer
wav_input_16khz = torch.randn(1, 10000)
if cfg.normalize:
    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
rep, layer_results = \
    model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
print(layer_reps)
