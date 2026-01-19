import numpy as np
import pandas as pd


def preprocess_dataset(
    dataset: pd.DataFrame, test: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select usable feature columns and produces log transformed features.

    Args:
        dataset (pd.DataFrame): DataFrame with feature columns and targets.
        test (bool, optional): Is dataset has target (test=False), or not.
        Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: If test, returns feature df twice,
        if train or val return features and targets.
    """
    dataset["item_condition"] = dataset["item_condition"].fillna("Нет")
    dataset_X = dataset[
        [
            "item_price",
            "item_condition",
            "category_name",
            "subcategory_name",
            "microcat_name",
        ]
    ]
    if not test:
        dataset["log_real_weight"] = np.log(dataset["real_weight"] + 1)
        dataset["log_real_height"] = np.log(dataset["real_height"] + 1)
        dataset["log_real_length"] = np.log(dataset["real_length"] + 1)
        dataset["log_real_width"] = np.log(dataset["real_width"] + 1)
        dataset_y = dataset[
            [
                "log_real_weight",
                "log_real_height",
                "log_real_length",
                "log_real_width",
            ]
        ]
        return dataset_X, dataset_y
    else:
        return dataset_X, dataset_X
