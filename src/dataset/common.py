
from typing import Optional, List, Dict
from pathlib import Path
import numpy as np

from src.utils.common import pad_if_needed
from src.config import InferenceConfig
from src.dataset.seg import SegTestDataset


def load_chunk_features(
        duration: int,
        feature_names: List[str],
        series_ids: Optional[List[str]],
        processed_dir: Path,
        phase: str,
) -> Dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir/phase).glob("*")]
    
    for series_id in series_ids:
        series_dir = processed_dir/phase/series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir/f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        for i in range(num_chunks):
            chunk_feature = this_feature[i*duration : (i+1)*duration]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            features[f"{series_id}_{i:07}"] = chunk_feature

        return features
    

def get_test_ds(
        cfg: InferenceConfig,
        chunk_features: Dict[str, np.ndarray],
) -> SegTestDataset:
    if cfg.dataset.name == "seg":
        return SegTestDataset(cfg=cfg, chunk_features=chunk_features)
    elif cfg.dataset.name == "detr":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}")