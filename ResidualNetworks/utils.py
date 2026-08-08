import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def threshold_to_index(values: np.ndarray) -> np.ndarray:
    """Converte valores contínuos de constelação em índices discretos de 4-PAM."""
    values = np.asarray(values)

    # Cada valor é mapeado para uma das quatro faixas separadas por -1, 0 e 1.
    return np.digitize(values, bins=[-1.0, 0.0, 1.0])


def qam16_gray_bits(values: np.ndarray) -> np.ndarray:
    """Converte valores 4-PAM em bits usando codificação Gray."""
    values = np.asarray(values)
    indices = threshold_to_index(values)

    # O mapa Gray garante que símbolos adjacentes diferem por apenas 1 bit.
    gray_map = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    return gray_map[indices]


def polarization_bits(points: np.ndarray) -> np.ndarray:
    """Converte pontos (I, Q) em pares de bits para cada amostra."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Esperado points shape (n_samples, 2)')
    return qam16_gray_bits(points)


def estimated_ber(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula uma estimativa de BER comparando símbolos verdadeiros e previstos."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if y_true.ndim != 2 or y_true.shape[1] != 4:
        raise ValueError('Expected y_true and y_pred shape (n_samples, 4)')

    # Separa as duas polarizações em X e Y.
    x_true = y_true[:, 0:2]
    y_true_pol = y_true[:, 2:4]
    x_pred = y_pred[:, 0:2]
    y_pred_pol = y_pred[:, 2:4]

    x_bits_true = polarization_bits(x_true)
    y_bits_true = polarization_bits(y_true_pol)
    x_bits_pred = polarization_bits(x_pred)
    y_bits_pred = polarization_bits(y_pred_pol)

    bit_errors = np.sum(x_bits_true != x_bits_pred) + np.sum(y_bits_true != y_bits_pred)
    total_bits = y_true.shape[0] * 8
    return float(bit_errors) / float(total_bits)


def window_sequence(x: np.ndarray, y: np.ndarray, n_sym: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria janelas de entrada e saída para treinar modelos com resíduo."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('x and y must be 2D arrays with channels first')
    if x.shape[0] != 4 or y.shape[0] != 4:
        raise ValueError('Expected 4 channels for dual-polarization data')

    n_elem = x.shape[1] - n_sym
    x_mult = []
    for i in range(n_sym):
        x_mult.append(x[0, i:n_elem + i])
        x_mult.append(x[1, i:n_elem + i])
        x_mult.append(x[2, i:n_elem + i])
        x_mult.append(x[3, i:n_elem + i])
    x_mult = np.array(x_mult)

    # Pega o símbolo central para comparação e o centro da janela de entrada.
    y_center = y[:, int(n_sym / 2): n_elem + int(n_sym / 2)]
    x_center = x[:, int(n_sym / 2): n_elem + int(n_sym / 2)]
    return x_center, x_mult, y_center


def split_train_test(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    """Divide sequências paralelas em conjuntos de treino e teste."""
    n_samples = x.shape[0]
    split_idx = int(n_samples * train_ratio)
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Aplica standard scaling aos dados de entrada."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler
