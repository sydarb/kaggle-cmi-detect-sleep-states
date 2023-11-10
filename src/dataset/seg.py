
from typing import Dict
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import numpy as np
import torch

from src.config import InferenceConfig
from src.utils.common import nearest_valid_size


class SegTestDataset(Dataset):
    def __init__(self,
        cfg: InferenceConfig,
        chunk_features: Dict[str, np.ndarray],
    ) -> None:
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }