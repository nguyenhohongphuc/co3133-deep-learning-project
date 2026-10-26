from .linear import LinearClassifier
from .mlp import MLP
from .cnn import SmallCNN
from .rnn import SequenceRNN
from .transformer import PatchTransformer



MODELS = {
    "linear": LinearClassifier,
    "mlp": MLP,
    "cnn": SmallCNN,
    "rnn": SequenceRNN,
    "transformer": PatchTransformer,
}