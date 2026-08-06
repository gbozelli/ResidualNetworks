import os
from typing import Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import load_dual_polarization_dataset
    from .utils import estimated_ber, standardize, split_train_test
except ImportError:
    from data_loader import load_dual_polarization_dataset
    from utils import estimated_ber, standardize, split_train_test


def build_direct_mlp(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', random_state=1) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=200,
        random_state=random_state,
        verbose=False,
    )


def prepare_direct_dataset(
    real_data: np.ndarray,
    ideal_data: np.ndarray,
    dbm_index: int = 0,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x = np.asarray(real_data[dbm_index]).T
    y = np.asarray(ideal_data).T
    x_train, x_test, y_train, y_test = split_train_test(x, y, train_ratio)
    x_train, x_test, x_scaler = standardize(x_train, x_test)
    y_train, y_test, y_scaler = standardize(y_train, y_test)
    return x_train, x_test, y_train, y_test, x_scaler, y_scaler


def train_and_evaluate_direct_mlp(
    real_data_dir: str,
    dbm_index: int = 0,
    distance: str = '175km',
    dbm_values: Optional[list] = None,
) -> None:
    dataset_dir = os.path.join(real_data_dir, distance)
    real_data, ideal_data = load_dual_polarization_dataset(
        real_data_dir,
        real_prefix='DP_RealConstellationDiagram_',
        ideal_filename=f'DP_IdealConstellationDiagram_0dbm_{distance}.csv',
        dbm_values=dbm_values,
    )

    x_train, x_test, y_train, y_test, x_scaler, y_scaler = prepare_direct_dataset(
        real_data,
        ideal_data,
        dbm_index=dbm_index,
    )

    model = build_direct_mlp()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    y_pred_original = y_scaler.inverse_transform(predictions)
    y_test_original = y_scaler.inverse_transform(y_test)

    ber = estimated_ber(y_test_original, y_pred_original)
    print(f'Direct MLP example completed for dBm index {dbm_index}')
    print(f'Estimated BER: {ber:.2e}')


def main() -> None:
    print('Direct MLP example for constellation regression')
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(
            'Data directory not found. Place constellation CSV files in the package `data/` folder.'
        )
    train_and_evaluate_direct_mlp(data_directory)


if __name__ == '__main__':
    main()
