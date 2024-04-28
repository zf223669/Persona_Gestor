from torch import nn
import torch
import librosa
from src import utils
from src.utils.wavlm.WavLM import WavLM, WavLMConfig
from os import path

log = utils.get_pylogger(__name__)


class WavLMEncoder(nn.Module):
    def __init__(self,
                 checkpoint_path='./wavlm/pretrain-models/WavLM-Large.pt'):
        super().__init__()
        # log.info(path.exists("/home/zf223669/Mount/Diffmotion-v3-sync/src/utils/wavlm/pretrain-models/WavLM-Base+.pt"))
        log.info(path.exists(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio_dataset):

        wav_input_16khz = audio_dataset
        # log.info('start wavlm encoding......')
        if self.cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(audio_dataset, audio_dataset.shape)
        # test = self.model.extract_features(wav_input_16khz)
        rep, layer_results = self.model.extract_features(wav_input_16khz, output_layer=self.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

        return rep, layer_reps

# class WavLMEncoder(WavLMPreTrainedModel):
#     def __init__(self,
#                  config):
#         super(WavLMEncoder, self).__init__(config)
#         self.wavlm = WavLMModel.from_pretrained(config)
#
#     def forward(self, input_values):
#         output = self.wavlm(input_values)
#         return output
#
# class WavLMEncoder(nn.Module):
#     def __init__(self,
#                  config,
#                  ):
#         super().__init__()
#
#         # self.processor = AutoProcessor.from_pretrained(config)
#         self.model = WavLMModel.from_pretrained(config)
#         for param in self.model.parameters():
#             param.requires_grad = False
#
#     def encode_audio(self, gesture_data):
#         return self.model(gesture_data)
#
#     def forward(self, audio_dataset, audio_sample_rate, calculateAtCPU=True):
#         # inputs = self.processor(audio_dataset, sampling_rate=audio_sample_rate, return_tensors="pt").to('cuda')
#         # log.info('Encoding gesture_data features with WavLM...')
#         with torch.no_grad():
#             if calculateAtCPU:
#                 if self.model.device != 'cpu':
#                     self.model.to('cpu')
#                 outputs = self.model(input_values=audio_dataset.to('cpu'))
#                 last_hidden_states = outputs.last_hidden_state  # [64,300,768][1,292,768]  60s=60000ms [1,2999,1024]
#             else:
#                 outputs = self.model(input_values=audio_dataset)
#                 last_hidden_states = outputs.last_hidden_state  # [64,300,768][1,292,768]  60s=60000ms [1,2999,1024]
#
#         # for name, param in self.model.named_parameters():
#         #     if param.requires_grad:
#         #         print(f'+++++++++++++++++++++++++++++++++++  requires_grad: {name}')
#         #     else:
#         #         print(f'----------------------------------  no requires_grad:{name}')
#
#         # list(last_hidden_states.shape)
#         # log.info('Done!!!')
#         return last_hidden_states
#
#
# if __name__ == "__main__":
#     model = WavLMEncoder()
#     y, sr = librosa.load('../../../data/TrinityDataSet/Sources/gesture_data/Recording_001.wav', sr=16000, duration=60)
#     output = model.forward(audio_dataset=y,audio_sample_rate=sr)
#     list(output.shape)
#     # torch.set_printoptions(profile="full",linewidth=500)
#     print(output.shape)
#     torch.cuda.empty_cache()
