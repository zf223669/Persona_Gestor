# Creating a decoder model
from src.diffmotion.torchscale.architecture.config import DecoderConfig
from src.diffmotion.torchscale.architecture.decoder import Decoder

config = DecoderConfig(vocab_size=64000)
decoder = Decoder(config)
print(decoder)

# Creating a encoder-decoder model
from src.diffmotion.torchscale.architecture.config import EncoderDecoderConfig
from src.diffmotion.torchscale.architecture.encoder_decoder import EncoderDecoder

config = EncoderDecoderConfig(vocab_size=64000)
encdec = EncoderDecoder(config)
print(encdec)