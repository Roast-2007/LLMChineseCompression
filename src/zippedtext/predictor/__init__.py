"""Predictors provide probability distributions for next-character prediction."""

from .base import Predictor
from .ngram import NgramPredictor

__all__ = ["Predictor", "NgramPredictor"]
