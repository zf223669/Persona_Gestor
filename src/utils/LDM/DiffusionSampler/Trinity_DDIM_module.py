from TrinityDiffusion.trinity_DDPM_module import TrinityDDPM
import pytorch_lightning as pl
from src import utils

log = utils.get_pylogger(__name__)


class TrinityDDIM(TrinityDDPM):
    def __init__(self,
                 test=1):
        super().__init__()
