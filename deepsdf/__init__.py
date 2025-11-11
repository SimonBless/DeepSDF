"""
DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation

This package provides a complete implementation of DeepSDF for learning implicit 
shape representations using signed distance functions.
"""

__version__ = "0.1.0"

from deepsdf.models.decoder import DeepSDFDecoder
from deepsdf.data.dataset import SDFDataset
from deepsdf.training.trainer import Trainer
from deepsdf.inference.reconstructor import ShapeReconstructor

__all__ = [
    "DeepSDFDecoder",
    "SDFDataset",
    "Trainer",
    "ShapeReconstructor",
]
