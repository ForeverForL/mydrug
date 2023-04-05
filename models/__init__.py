from .classifier import *
from .generator import *
from .rlearner import *
from .encoderdecoder import *
from .attention import *
from .interfaces import *
from explorer import *

DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEFAULT_GPUS = (0,)