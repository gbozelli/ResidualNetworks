import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import find_constellation_files, load_dual_polarization_dataset, plot_constellation
    from .utils import estimated_ber
except ImportError:
    from data_loader import find_constellation_files, load_dual_polarization_dataset, plot_constellation
    from utils import estimated_ber


# Níveis de amplitude usados para gerar símbolos 16-QAM.
QAM_LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)


def generate_random_16qam_symbols(n_symbols: int) -> np.ndarray:
    """Gera símbolos 16-QAM aleatórios para duas polarizações."""
    x_iq = np.random.choice(QAM_LEVELS, size=(n_symbols, 2))
    y_iq = np.random.choice(QAM_LEVELS, size=(n_symbols, 2))

    # Combina os dados das duas polarizações em uma única matriz.
    return np.hstack([x_iq, y_iq])


def add_awgn_noise(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    """Adiciona ruído branco gaussiano ao sinal em função da relação sinal-ruído."""
    power = np.mean(symbols ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)

    return symbols + np.random.normal(scale=noise_std, size=symbols.shape)


def load_real_example() -> tuple[np.ndarray, np.ndarray] | None:
    """Tenta carregar um exemplo real de constelação do diretório data/."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    real_files = find_constellation_files(data_dir, 'DP_RealConstellationDiagram_*.csv')
    ideal_path = os.path.join(data_dir, 'DP_IdealConstellationDiagram_0dbm.csv')

    if not real_files:
        return None
    if not os.path.isfile(ideal_path):
        return None

    real_data, ideal_data = load_dual_polarization_dataset(
        data_dir,
        real_prefix='DP_RealConstellationDiagram_',
        ideal_filename='DP_IdealConstellationDiagram_0dbm.csv',
        dbm_values=[0],
        skiprows=5,
    )

    x = real_data[0]
    y = ideal_data

    if x.shape[0] == 8:
        print('Detectado arquivo de constelação com 8 linhas. Usando linhas 5-8 como sinal recebido.')
        x = x[4:8, :]
    else:
        print('Usando sinal de constelação com 4 linhas.')

    return x.T, y.T


def train_simple_mlp(x_train: np.ndarray, y_train: np.ndarray) -> tuple[MLPRegressor, StandardScaler]:
    """Cria e treina um MLP simples usando os dados de treinamento."""
    scaler = StandardScaler()

    # Escala as entradas para média 0 e variância 1.
    x_train_scaled = scaler.fit_transform(x_train)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=150,
        random_state=1,
        verbose=False,
    )

    model.fit(x_train_scaled, y_train)
    return model, scaler


def show_constellation(symbols: np.ndarray, title: str) -> None:
    """Mostra o diagrama de constelação usando apenas os dois primeiros componentes do sinal."""
    plot_constellation(symbols[:, 0], symbols[:, 1], title=title)


def run_synthetic_example() -> None:
    """Executa um exemplo sintético de 16-QAM com ruído e estima o BER."""
    print('Executando exemplo sintético de 16-QAM...')
    symbols = generate_random_16qam_symbols(2000)
    noisy_symbols = add_awgn_noise(symbols, snr_db=12)

    show_constellation(noisy_symbols, 'Constelação 16-QAM ruidosa (polarização X)')

    x_train, x_test, y_train, y_test = train_test_split(noisy_symbols, symbols, test_size=0.2, random_state=1)
    model, scaler = train_simple_mlp(x_train, y_train)

    y_pred = model.predict(scaler.transform(x_test))
    ber = estimated_ber(y_test, y_pred)

    print('Exemplo sintético completado.')
    print(f'Forma dos dados de entrada: {symbols.shape}')
    print(f'BER estimado no conjunto de teste: {ber:.4e}')


def run_real_example(x: np.ndarray, y: np.ndarray) -> None:
    """Executa o exemplo usando dados reais de constelação carregados do arquivo."""
    print('Executando exemplo com dados reais de constelação...')

    # Usa apenas as primeiras 2000 amostras para não deixar a execução muito pesada.
    x = x[:2000, :]
    y = y[:2000, :]

    show_constellation(x, 'Constelação real recebida (polarização X)')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model, scaler = train_simple_mlp(x_train, y_train)

    y_pred = model.predict(scaler.transform(x_test))
    ber = estimated_ber(y_test, y_pred)

    print('Exemplo com dados reais completado.')
    print(f'Forma dos dados carregados: {x.shape}')
    print(f'BER estimado no conjunto de teste: {ber:.4e}')


def main() -> None:
    """Função principal que escolhe entre dados reais ou exemplo sintético."""
    print('Tutorial didático: exemplo de 16-QAM, MLP direto e BER.')
    real_data = load_real_example()
    if real_data is not None:
        x_real, y_real = real_data
        run_real_example(x_real, y_real)
    else:
        print('Não foram encontrados dados reais em data/. Executando exemplo sintético.')
        run_synthetic_example()

    print('Tutorial finalizado.')


if __name__ == '__main__':
    main()
