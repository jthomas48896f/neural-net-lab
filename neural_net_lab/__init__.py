"""
Neural Network Laboratory - A from-scratch neural network library.

Built entirely on numpy with interactive visualizations using matplotlib.
Implements full backpropagation, multiple optimizers, and learning rate scheduling.
"""

from .tensor import Tensor
from .layers import Dense, Dropout, BatchNorm, Parameter
from .activations import (
    ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, get_activation
)
from .losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, get_loss
from .optimizers import SGD, Adam, RMSProp, AdaGrad, StepDecay, CosineAnnealing
from .network import Network, TrainingHistory, PrintProgress, EarlyStopping
from .transformer import (
    LayerNorm, RMSNorm, RotaryPositionalEmbedding,
    MultiHeadSelfAttention, GroupedQueryAttention, FeedForward,
    SwiGLUFeedForward, TransformerBlock, CharTransformerLM,
)
from .datasets import (
    make_spirals, make_moons, make_circles, make_xor,
    make_gaussian_blobs, make_mnist_subset
)

__version__ = "1.0.0"
__all__ = [
    "Tensor",
    "Dense", "Dropout", "BatchNorm", "Parameter",
    "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax", "GELU",
    "MSELoss", "CrossEntropyLoss", "BinaryCrossEntropyLoss",
    "SGD", "Adam", "RMSProp", "AdaGrad", "StepDecay", "CosineAnnealing",
    "Network", "TrainingHistory", "PrintProgress", "EarlyStopping",
    "LayerNorm", "RMSNorm", "RotaryPositionalEmbedding",
    "MultiHeadSelfAttention", "GroupedQueryAttention", "FeedForward",
    "SwiGLUFeedForward", "TransformerBlock",
    "CharTransformerLM",
    "make_spirals", "make_moons", "make_circles", "make_xor",
    "make_gaussian_blobs", "make_mnist_subset",
]
