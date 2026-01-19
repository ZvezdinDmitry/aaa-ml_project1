import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)


def print_metrics(y_target: pd.DataFrame, y_preds: np.ndarray) -> None:
    """Prints MAE, R^2, pearson corr, and mean MAE.

    Args:
        y_target (pd.DataFrame): Target features DF
        y_preds (np.ndarray): Predicted features array,
        must be the same size as y_target.
    """
    mae_list = []
    for i, col in enumerate(y_target.columns):
        mae = mean_absolute_error(y_target[col], y_preds[:, i])
        r2 = r2_score(y_target[col], y_preds[:, i])
        corr, _ = pearsonr(y_target[col], y_preds[:, i])
        mae_list.append(mae)
        print(f"{col}:\n\t{mae=:.04f}\n\t{r2=:.04f}\n\t{corr=:.04f}")
    print(f"Mean MAE: {np.mean(mae_list):.04f}")
