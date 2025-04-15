
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

LABEL_MAPPER = {'Cars': 0, 'Drones': 1, 'People': 2}


def get_data_for_label(label, data_dir_path: str):
    X, y = [], []
    for root, dirs, files in os.walk(os.path.join(data_dir_path, label)):
        for file in files:
            if file.endswith('.csv'):
                y.append(LABEL_MAPPER[label])
                df = pd.read_csv(os.path.join(root, file), sep=',', header=None)
                X.append(df.values)
    print(f'Loaded {len(y)} examples for label {label} encoded with {LABEL_MAPPER[label]}')
    return X, y

def main():
    data_dir_path: Path = Path(data_dir_path_str)
    assert data_dir_path.is_dir(), f"{data_dir_path} is not a directory."

    X_cars, y_cars = get_data_for_label('Cars', str(data_dir_path))
    X_drones, y_drones = get_data_for_label('Drones', str(data_dir_path))
    X_people, y_people = get_data_for_label('People', str(data_dir_path))

    X = X_cars + X_drones + X_people
    y = y_cars + y_drones + y_people
    X, y = np.array(X), np.array(y)

if __name__ == '__main__':
    load_dotenv()

    data_dir_path_str: str = os.getenv('DATA_DIR')
    assert data_dir_path_str is not None, f"{data_dir_path_str} is None."

    main()
