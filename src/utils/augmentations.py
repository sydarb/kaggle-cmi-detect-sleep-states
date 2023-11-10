from typing import Any, Tuple

import torch
import numpy as np


def get_rand_1d_bbox(n_timesteps: int, gamma: float) -> Tuple[int, int]:
    cut_ratio = np.sqrt(1.0 - gamma)
    cut_length = int(n_timesteps * cut_ratio)
    start = np.random.randint(0, n_timesteps - cut_length)
    end = start + cut_length
    return start, end


class Mixup:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha

    def __call__(self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = imgs.size(0)
        idx = torch.randperm(batch_size)

        gamma = np.random.beta(self.alpha, self.alpha)
        mixed_imgs: torch.Tensor = gamma*imgs + (1-gamma)*imgs[idx] 
        mixed_labels: torch.Tensor = gamma*labels + (1-gamma)*labels[idx]

        return mixed_imgs, mixed_labels
    

class Cutmix:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha

    def __call__(self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = imgs.size(0)
        idx = torch.randperm(batch_size)

        shuffled_imgs = imgs[idx]
        shuffled_labels = labels[idx]
        gamma = np.random.beta(self.alpha, self.alpha)

        start, end = get_rand_1d_bbox(imgs.size(2), gamma)
        mixed_imgs = torch.concatenate(
            [imgs[:, :, :start], shuffled_imgs[:, :, start:end], imgs[:, :, end:]], dim=2)
        mixed_labels = torch.concatenate(
            [labels[:, :start, :], shuffled_labels[:, start:end, :], labels[:, end:, :]], dim=1)
        
        return mixed_imgs, mixed_labels