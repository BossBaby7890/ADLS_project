from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import List, Sequence

import torch
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


class CalibrationSampleSelector:
    """
    Build a smaller calibration subset from a dataset.

    Supported strategies:
    - random
    - class_balanced
    """

    def __init__(
        self,
        strategy: str = "random",
        num_samples: int = 256,
        seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.num_samples = num_samples
        self.seed = seed

    def select(self, dataset) -> Subset:
        if self.strategy == "random":
            indices = self._random_indices(len(dataset))
        elif self.strategy == "class_balanced":
            indices = self._class_balanced_indices(dataset)
        else:
            raise ValueError(f"Unknown calibration selection strategy: {self.strategy}")

        logger.info(
            "Calibration selector | strategy=%s | selected=%d",
            self.strategy,
            len(indices),
        )
        return Subset(dataset, indices)

    def _random_indices(self, n: int) -> List[int]:
        rng = random.Random(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)
        return indices[: min(self.num_samples, n)]

    def _class_balanced_indices(self, dataset) -> List[int]:
        label_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label_to_indices[int(label)].append(idx)

        classes = sorted(label_to_indices.keys())
        if not classes:
            return []

        per_class = max(1, self.num_samples // len(classes))
        rng = random.Random(self.seed)

        chosen = []
        for cls in classes:
            cls_indices = label_to_indices[cls]
            rng.shuffle(cls_indices)
            chosen.extend(cls_indices[:per_class])

        # top up if needed
        if len(chosen) < self.num_samples:
            remaining = []
            chosen_set = set(chosen)
            for idx in range(len(dataset)):
                if idx not in chosen_set:
                    remaining.append(idx)
            rng.shuffle(remaining)
            chosen.extend(remaining[: self.num_samples - len(chosen)])

        return chosen[: self.num_samples]

