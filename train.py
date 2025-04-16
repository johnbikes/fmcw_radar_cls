import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from nn_utils import (
    MapsDataset,
    Conv2Net,
    Conv1Net,
    MLPNet,
    ConvSimpleNet,
    train_validate,
    plot_train_val_loss,
    train_model,
    set_seed,
)
from load_data import get_data_for_label


def main():
    data_dir_path: Path = Path(data_dir_path_str)
    assert data_dir_path.is_dir(), f"{data_dir_path} is not a directory."

    X_cars, y_cars = get_data_for_label('Cars', str(data_dir_path))
    X_drones, y_drones = get_data_for_label('Drones', str(data_dir_path))
    X_people, y_people = get_data_for_label('People', str(data_dir_path))

    X = X_cars + X_drones + X_people
    y = y_cars + y_drones + y_people
    X, y = np.array(X), np.array(y)

    # Data normalization to normal distribution
    def standard_normalizer(X):
        n_samples = X.shape[0]
        for sid in range(n_samples):
            matrix = X[sid, :, :]
            X[sid, :, :] = (matrix - np.mean(matrix)) / np.std(matrix)
        return X

    X = standard_normalizer(X)
    n_total = len(X)

    # separating train, validation, test
    test_size = 0.1
    val_size = 0.2
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size,
        # random_state=SEED, 
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size / (1 - test_size),
        # random_state=SEED,
        stratify=y_trainval,
    )
    train_dataset = MapsDataset(X_train, y_train)
    val_dataset = MapsDataset(X_val, y_val)

    print(f"Training: {len(X_train)} samples -> {len(X_train)/n_total*100:.2f} %")
    print(f"Validation: {len(X_val)} samples -> {len(X_val)/n_total*100:.2f} %")
    print(f"Test: {len(X_test)} samples -> {len(X_test)/n_total*100:.2f} %")
    """
    Training: 12239 samples -> 70.00 %
    Validation: 3497 samples -> 20.00 %
    Test: 1749 samples -> 10.00 %
    """

    # set_seed(SEED+3)
    batch_size = 32
    lr = 2e-4
    num_epochs = 25
    k1_size = (3, 3)
    p_dropout = 0.
    model = Conv1Net(k1_size)
    train_loss, val_loss, train_acc, val_acc, model = train_model(
        model, train_dataset, val_dataset, batch_size, num_epochs, lr
    )
    fig = plot_train_val_loss(
        num_epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        title=f"{type(model).__name__} | lr={lr} | bs={batch_size}",
    )
    plt.tight_layout()
    fig.savefig('./conv1_results.png')


if __name__ == "__main__":
    load_dotenv()

    data_dir_path_str: str = os.getenv('DATA_DIR')
    assert data_dir_path_str is not None, f"{data_dir_path_str} is None."

    main()