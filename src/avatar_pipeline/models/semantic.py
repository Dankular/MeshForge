from dataclasses import dataclass
import numpy as np


@dataclass
class SemanticMap:
    labels: np.ndarray
    confidence: np.ndarray
    palette: dict[int, str]
    alpha: np.ndarray | None = None


@dataclass
class PoseData:
    joint_names: list[str]
    keypoints: np.ndarray
