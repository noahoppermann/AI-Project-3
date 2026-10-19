import math

from model import Model


def train(examples):
    """TRAIN. examples is a list of (words, label) pairs."""
    raise NotImplementedError


def prior(model, c):
    """PRIOR."""
    raise NotImplementedError


def likelihood(model, w, c, k=1):
    """LIKELIHOOD."""
    raise NotImplementedError


def log_score(model, words, c, k=1):
    """LOG-SCORE."""
    raise NotImplementedError


def classify(model, words, k=1):
    """CLASSIFY."""
    raise NotImplementedError


def accuracy(model, examples, k=1):
    """ACCURACY."""
    raise NotImplementedError
