from .afno import AFNO
from .avit import AViT
from .dilated_resnet import DilatedResNet
from .fno import FNO
from .refno import ReFNO
from .tfno import TFNO
from .unet_classic import UNetClassic
from .unet_convnext import UNetConvNext
from .cvit import CViT
from .transolver import Transolver_
from .sinenet import sinenet
from .ffno import FFNO

__all__ = [
    "FNO",
    "TFNO",
    "UNetClassic",
    "UNetConvNext",
    "DilatedResNet",
    "ReFNO",
    "AViT",
    "AFNO",
    "CViT",
    "Transolver_",
    "sinenet",
    "FFNO",
]
