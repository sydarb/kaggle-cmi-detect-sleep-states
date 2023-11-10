from typing import List

from scipy.signal import find_peaks
import numpy as np
import polars as pl


def post_process_for_seg(
        keys: List[str],
        preds: np.ndarray,
        score_th: float = 0.01,
        distance: int = 5000,
) -> pl.DataFrame:
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]
            for step, score in zip(steps, scores):
                records.append({
                    "series_id": series_id,
                    "step": step,
                    "event": event_name,
                    "score": score,
                })
    
    if len(records) == 0:
        records.append({
            "series_id": series_id,
            "step": 0,
            "event": "onset",
            "score": 0,
        })
    
    submit_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(submit_df)))
    submit_df = submit_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])

    return submit_df