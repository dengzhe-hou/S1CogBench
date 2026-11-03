"""Centaur paper-form weighted-additive baseline, four shared free weights.

This is a transparent reimplementation of the published equation, not a claim
that the missing original task-specific training entry point was recovered.
"""
import re

import torch


def visible_features(trial):
    products = re.findall(r'Product ([A-Z]) ratings: \[([01] [01] [01] [01])\]', trial.stimulus)
    if tuple(key for key, _ in products) != trial.output_keys:
        raise ValueError('Feature order must equal the two original product keys.')
    # Each dimension retains the original displayed expert order. Validities
    # are not multiplied in: the published model fits four unconstrained w's.
    return [[int(value) for value in values.split()] for _, values in products]


class WeightedAdditive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))

    def forward(self, features):
        return features @ self.weights
