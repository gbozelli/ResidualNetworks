import os
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableMating, MixedVariableSampling
    from pymoo.core.problem import Problem
    from pymoo.core.variable import Integer
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError:  # pragma: no cover
    NSGA2 = None
    MixedVariableDuplicateElimination = None
    MixedVariableMating = None
    MixedVariableSampling = None
    Problem = None
    Integer = None
    minimize = None

try:
    from .data_loader import load_dual_polarization_dataset
    from .utils import estimated_ber, standardize, window_sequence
except ImportError:
    from data_loader import load_dual_polarization_dataset
    from utils import estimated_ber, standardize, window_sequence


def build_resnet_model(input_dim: int, hidden_layers=(128, 64)) -> Model:
    """Cria um modelo MLP com conexão residual (ResNet simples)."""
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)

    # A saída residual tem 4 valores, correspondendo ao sinal dual-polarization.
    residual = layers.Dense(4, activation='linear')(x)
    if input_dim != 4:
        # Se a entrada não tiver 4 componentes, projeta-se a entrada para 4.
        x_proj = layers.Dense(4, activation='linear')(inputs)
        outputs = layers.Add()([residual, x_proj])
    else:
        outputs = layers.Add()([residual, inputs])

    model = Model(inputs=inputs, outputs=outputs, name='ResNetMLP')
    model.compile(optimizer='adam', loss='mse')
    return model


def model_flops(input_dim: int, hidden_layers: Tuple[int, ...]) -> float:
    """Calcula uma estimativa simples do número de operações do modelo (FLOPs)."""
    layers_sizes = [input_dim] + list(hidden_layers) + [4]
    flops = 0
    for i in range(len(layers_sizes) - 1):
        flops += layers_sizes[i] * layers_sizes[i + 1] * 2
    return float(flops)


def prepare_resnet_training(
    real_signal: np.ndarray,
    ideal_signal: np.ndarray,
    n_sym: int = 3,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepara os dados de treino do modelo ResNet a partir da sequência real e ideal."""
    x_center, x_window, y_center = window_sequence(real_signal, ideal_signal, n_sym)

    # O modelo aprende o resíduo entre o símbolo ideal e o símbolo central.
    y_residual = (y_center - x_center).T
    X = x_window.T
    y = y_residual

    X_train, X_test, y_train, y_test = np.split(X, [int(len(X) * train_ratio)], axis=0) + np.split(y, [int(len(y) * train_ratio)], axis=0)
    x_train_scaled, x_test_scaled, scaler = standardize(X_train, X_test)
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, x_center.T


def objective_resnet(
    real_signal: np.ndarray,
    ideal_signal: np.ndarray,
    hidden_layers=(128, 64),
    n_sym: int = 3,
    max_epochs: int = 20,
) -> float:
    """Avalia um conjunto de parâmetros de rede ResNet retornando a BER do modelo."""
    x_train, x_test, y_train, y_test, _, x_center = prepare_resnet_training(
        real_signal, ideal_signal, n_sym=n_sym
    )
    model = build_resnet_model(x_train.shape[1], hidden_layers)
    model.fit(x_train, y_train, epochs=max_epochs, batch_size=128, verbose=0)
    preds = model.predict(x_test, verbose=0)
    recovered = preds + x_center[: len(preds)]
    ber = estimated_ber(y_test + x_center[: len(preds)], recovered)
    return ber


def execute_nsga2(
    real_signal: np.ndarray,
    ideal_signal: np.ndarray,
    dbm: int = 8,
    pop_size: int = 12,
    n_gen: int = 6,
) -> Optional[Dict[str, np.ndarray]]:
    if NSGA2 is None:
        raise ImportError('pymoo is required to run NSGA2 optimization. Install via pip install pymoo')

    variables = {
        'layers': Integer(bounds=(1, 2)),
        'l1': Integer(bounds=(32, 256)),
        'l2': Integer(bounds=(32, 256)),
        'n_sym': Integer(bounds=(1, 11)),
    }

    class ResNetProblem(Problem):
        def __init__(self):
            super().__init__(vars=variables, n_obj=2)

        def _evaluate(self, X, out, *args, **kwargs):
            """Avalia cada indivíduo na população do NSGA2."""
            F = np.zeros((len(X), 2))
            for i, individual in enumerate(X):
                n_sym = int(individual['n_sym'])
                hidden_layers = []
                if individual['layers'] >= 1:
                    hidden_layers.append(int(individual['l1']))
                if individual['layers'] >= 2:
                    hidden_layers.append(int(individual['l2']))
                hidden_layers = tuple(hidden_layers) if hidden_layers else (64,)

                try:
                    ber = objective_resnet(real_signal, ideal_signal, hidden_layers, n_sym)
                except Exception:
                    # Em caso de falha no modelo, usamos BER alta para descartar a solução.
                    ber = 1.0
                flops = model_flops(4 * n_sym, hidden_layers)
                F[i, :] = [ber, flops]
            out['F'] = F

    termination = get_termination('n_gen', n_gen)
    problem = ResNetProblem()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )
    result = minimize(problem, algorithm, termination, seed=1, verbose=False)
    return {'X': result.X, 'F': result.F}


def main() -> None:
    print('NSGA2 + ResNet neural network example')
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(
            'Data directory not found. Place constellation CSV files in the package `data/` folder.'
        )
    print('Please adapt this file to load your constellation data before running NSGA2.')


if __name__ == '__main__':
    main()
