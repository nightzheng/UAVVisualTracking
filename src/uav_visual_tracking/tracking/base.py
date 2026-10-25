from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Prediction:
    tracks: np.ndarray  # T,N,2 in processing pixels
    visibility: np.ndarray  # T,N bool prediction, not ground truth
    metadata: dict
    # Optional continuous model outputs, T,N; legacy three-argument construction stays valid.
    raw_scores: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self, frame_count, point_count):
        if self.tracks.shape != (frame_count, point_count, 2) or self.visibility.shape != (frame_count, point_count):
            raise ValueError("Invalid prediction shapes")
        if self.visibility.dtype != np.bool_:
            raise ValueError("Visibility contract requires bool; do not fabricate confidence")
        for name, values in self.raw_scores.items():
            if not name.isidentifier() or values.shape != (frame_count, point_count) or values.dtype.kind != 'f':
                raise ValueError("Raw scores must be named floating T,N arrays")


class Tracker(ABC):
    @abstractmethod
    def predict(self, clip, queries) -> Prediction:
        """Return tracks, predicted visibility, identity and temporal permissions."""
