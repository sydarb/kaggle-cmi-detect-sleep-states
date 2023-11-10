
from typing import Union, Any

from src.models.base import BaseModel
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.decoder.unet import UNet1DDecoder
from src.models.spec2dcnn import Spec2DCNN
from src.config import InferenceConfig, FeatExtrConfig, DecoderConfig


def get_feature_extractor(
        cfg: FeatExtrConfig,
        feature_dims: int,
        num_timesteps: int,
) -> CNNSpectrogram:
    if cfg.name == "CNNSpectogram":
        feature_extractor = CNNSpectrogram(
            in_channels = feature_dims,
            output_size = num_timesteps,
            **cfg.params
        )
    elif cfg.name == "PANNsFeatureExtractor":
        raise NotImplementedError
    elif cfg.name == "LSTMFeatureExtractor":
        raise NotImplementedError
    elif cfg.name == "SpecFeatureExtractor":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.name}")
    
    return feature_extractor


def get_decoder(
        cfg: DecoderConfig,
        num_channels: int,
        num_classes: int,
        num_timesteps: int,
) -> UNet1DDecoder:
    if cfg.name == "Unet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels = num_channels,
            n_classes = num_classes,
            duration = num_timesteps,
            **cfg.params
        )
    elif cfg.name == "LSTMDecoder":
        raise NotImplementedError
    elif cfg.name == "TransformerDecoder":
        raise NotImplementedError
    elif cfg.name == "MLPDecoder":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid decoder name: {cfg.name}")
    
    return decoder


def get_model(
        cfg: InferenceConfig,
        feature_dims: int,
        num_classes: int,
        num_timesteps: int,
        test: bool = False,
) -> BaseModel:
    if cfg.model.name == "Spec2DCNN":
        feature_extractor = get_feature_extractor(
            cfg=cfg.feature_extractor,
            feature_dims=feature_dims,
            num_timesteps=num_timesteps
        )
        decoder = get_decoder(
            cfg=cfg.decoder,
            num_channels=feature_extractor.height,
            num_classes=num_classes,
            num_timesteps=num_timesteps
        )
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            encoder_weights=cfg.model.params["encoder_weights"] if test else None,
            encoder_name = cfg.model.params["encoder_name"]
        )
    elif cfg.model.name == "Spec1D":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid Model Name: {cfg.model.name}")
    
    return model