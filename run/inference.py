from pathlib import Path
from typing import List, Tuple

import torch
import hydra
import numpy as np
import polars as pl
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import InferenceConfig
from src.models.base import BaseModel
from src.models.common import get_model
from src.dataset.common import load_chunk_features, get_test_ds
from src.utils.common import seed_everything, nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg


def load_model(
        cfg: InferenceConfig
    ) -> BaseModel:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model: BaseModel = get_model(
        cfg=cfg,
        feature_dims=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps//cfg.downsample_rate,
        test=True
    )
    # load weights
    if cfg.weight is not None:
        weight_path = (Path(cfg.dir.model_dir)/cfg.weight.exp_name/cfg.weight.run_name/"best_model.pth")
        model.load_state_dict(torch.load(weight_path))
        print(f"loaded model weights from: {weight_path}")
    
    return model


def get_test_dataloader(
        cfg: InferenceConfig
    ) -> DataLoader:
    feature_dir = Path(cfg.dir.processed_dir)/cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def predict(
        duration: int,
        data_loader: DataLoader,
        model: BaseModel,
        device: torch.device,
        use_amp: bool 
    ) -> Tuple[List[str], np.ndarray]:
    model = model.to(device)
    model.eval()

    preds, keys = [], []
    for batch in tqdm(data_loader, desc="Inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                output = model.predict(x, orig_duration=duration)
                if output.preds is None:
                    raise ValueError("output.preds is None")
                key = batch["key"]
                preds.append(output.preds.detach().cpu().numpy())
                keys.extend(key)
    preds = np.concatenate(preds)

    return keys, preds


def make_submission(
        keys: List[str],
        preds: np.ndarray,
        downsample_rate: int,
        score_th: float,
        distance: int,
) -> pl.DataFrame:
    submit_df = post_process_for_seg(
        keys=keys,
        preds=preds,
        downsample_rate=downsample_rate,
        score_th=score_th,
        distance=distance
    )
    return submit_df


@hydra.main(config_path="config", config_name="inference")
def main(cfg: InferenceConfig) -> None:
    seed_everything(cfg.seed)
    print(f"Using global seed: {cfg.seed}")
    with trace("Load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("Load model"):
        model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with trace("Make prediction"):
        keys, preds = predict(cfg.duration, test_dataloader, model, device, cfg.use_amp)
    with trace("Make submission"):
        submit_df = make_submission(keys, preds, cfg.downsample_rate, 
            cfg.post.score_th, cfg.post.distance)
    submit_df.write_csv(Path(cfg.dir.submit_dir)/"submission.csv")


if __name__ == "__main__":
    main()