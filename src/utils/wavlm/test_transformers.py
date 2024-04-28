import os.path

from transformers import AutoProcessor, WavLMModel
import torch
from datasets import load_dataset
import librosa

print(librosa.show_versions())
# print(os.path.exists('../Audio/Recording_001.wav'))
# filename = librosa.ex('../Audio/Recording_001')
y, sr = librosa.load('../Audio/Recording_001.wav', sr=16000, duration=60)  # y shape :11553049
# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["gesture_data"].sampling_rate  # 16 khz

# processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
# model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")
model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")

# gesture_data file is decoded on the fly
# getted_dataset = dataset[0]["gesture_data"]["array"] # 93680
getted_dataset = y
inputs = processor(getted_dataset, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state  # [1,292,768]  60s=60000ms [1,2999,1024]
list(last_hidden_states.shape)
# torch.set_printoptions(profile="full",linewidth=500)

print(outputs)
