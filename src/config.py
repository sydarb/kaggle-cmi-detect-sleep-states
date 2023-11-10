from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str
    model_dir: str
    submit_dir: str

@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_workers: int
    offset: int
    sigma: int
    bg_sampling_rate: float

@dataclass
class PrepareDataConfig:
    dir: DirConfig
    phase: str

@dataclass
class AugmentationConfig:
    mixup_prob: float
    mixup_alpha: float
    cutmix_prob: float
    cutmix_alpha: float

@dataclass
class WeightConfig:
    exp_name: str
    run_name: str

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]

@dataclass
class FeatExtrConfig:
    name: str
    params: Dict[str, Any]

@dataclass
class DecoderConfig:
    name: str
    params: Dict[str, Any]

@dataclass
class PostProcessingConfig:
    score_th: float
    distance: int

@dataclass
class InferenceConfig:
    exp_name: str
    phase: str
    seed: int
    batch_size: int
    num_workers: int
    duration: int
    downsample_rate: int
    upsample_rate: int
    use_amp: bool
    labels: List[str]
    features: List[str]
    dir: DirConfig
    model: ModelConfig
    feature_extractor: FeatExtrConfig
    decoder: DecoderConfig
    weight: WeightConfig
    dataset: DatasetConfig
    aug: AugmentationConfig
    post: PostProcessingConfig