
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.transforms.functional import resize 

from src.models.base import BaseModel
from src.utils.augmentations import Mixup, Cutmix

class Spec2DCNN(BaseModel):
    def __init__(self,
            feature_extractor: nn.Module,
            decoder: nn.Module,
            encoder_name: str,
            in_channels: int,
            encoder_weights: Optional[str] = None,
            mixup_alpha: float = 0.5,
            cutmix_alpha: float = 0.5
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1
        )
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _forward(self, 
            x: torch.Tensor, 
            labels: Optional[torch.Tensor] = None, 
            do_mixup: bool = False, 
            do_cutmix: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.feature_extractor(x)   # (batch_size, n_channels, height, n_timesteps)
        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)
        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        logits = self.decoder(x)        # (batch_size, n_timesteps, n_classes)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def logits_to_probs_per_step(self, 
            logits: torch.Tensor, 
            orig_duration: int
    ) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[orig_duration, preds.shape[-1]], antialias=False)
    
    def correct_labels(self, 
            labels: torch.Tensor, 
            orig_duration: int
    ) -> torch.Tensor:
        return resize(labels, size=[orig_duration, labels.shape[-1]], antialias=False)