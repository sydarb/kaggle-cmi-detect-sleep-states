
from typing import Union, Optional, Tuple
from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    preds: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class BaseModel(nn.Module):
    def forward(self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            do_mixup: bool = False,
            do_cutmix: bool = False,
    ) -> ModelOutput:
        """forward pass of the model"""
        if labels is not None:
            logits, labels = self._forward(x, labels, do_mixup, do_cutmix)
            loss = self.loss_fn(logits, labels)
            return ModelOutput(logits, loss)
        else:
            logits = self._forward(x, labels=None, do_mixup=False, do_cutmix=False)
            if isinstance(logits, torch.Tensor):
                return ModelOutput(logits)
            else:
                raise ValueError("logits must be a torch.Tensor")

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        orig_duration: int,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        output = self.forward(x, labels, False, False)
        preds = self.logits_to_probs_per_step(output.logits, orig_duration)
        output.preds = preds

        if labels is not None:
            labels = self.correct_labels(labels, orig_duration)
            output.labels = labels

        return output

    @abstractmethod
    def _forward(self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            do_mixup: bool = False,
            do_cutmix: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def loss_fn(self,
            logits: torch.Tensor,
            labels: torch.Tensor
        ) -> torch.Tensor:
        """Calculate loss"""
        raise NotImplementedError 

    @abstractmethod
    def logits_to_probs_per_step(self,
            logits: torch.Tensor,
            orig_duration: int,
    ) -> torch.Tensor:
        """Convert logits to probabilities per step."""
        raise NotImplementedError

    @abstractmethod
    def correct_labels(self,
            labels: torch.Tensor,
            orig_duration: int,
    ) -> torch.Tensor:
        """Correct labels to match the output of the model."""
        raise NotImplementedError