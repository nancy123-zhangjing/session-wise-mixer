"""Pre-registered checkpoint selection for L5O protocol v2 (validation only)."""
from __future__ import annotations

import math

RULE = 'smoothed3'


def smoothed_choice(points):
    """Pick the node with the best 3-point centred moving average of validation scores.

    points: [(node, score), ...] in training order (every validated node, none skipped).
    Interior nodes average (previous, self, next); the two end nodes average the
    available neighbours only. Ties keep the EARLIER node. Returns
    (node, smoothed_score, raw_score, all_smoothed).
    """
    if not points:
        raise ValueError('No validation points to select from')
    scores = [float(s) for _, s in points]
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('Selection requires finite validation scores')
    smoothed = []
    for i in range(len(scores)):
        window = scores[max(0, i - 1):i + 2]
        smoothed.append(sum(window) / len(window))
    best = max(range(len(points)), key=lambda i: (smoothed[i], -i))
    return points[best][0], smoothed[best], scores[best], smoothed


def raw_choice(points):
    """Single best validation node; ties keep the earlier node (historical rule)."""
    if not points:
        raise ValueError('No validation points to select from')
    best = max(range(len(points)), key=lambda i: (float(points[i][1]), -i))
    return points[best][0], float(points[best][1])
