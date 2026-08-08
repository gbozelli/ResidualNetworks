import os
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_csv_matrix(path: str, sep: str = ',', skiprows: int = 0) -> np.ndarray:
    """Carrega um arquivo CSV e retorna os dados em formato polarization-first.

    O formato channel-first deixa os canais nas linhas e as amostras nas colunas.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")

    # Lê o CSV em uma matriz NumPy.
    data = pd.read_csv(path, sep=sep, skiprows=skiprows).values

    # Transpõe os dados para que cada linha represente uma polarização.
    return np.transpose(data)


def load_dp_constellations(
    directory: str,
    prefix: str,
    dbm_values: List[int],
    suffix: str = '.csv',
    skiprows: int = 5,
) -> np.ndarray:
    """Carrega vários arquivos de constelação dual-polarization para diferentes níveis de potência."""
    samples = []
    for dbm in dbm_values:
        path = os.path.join(directory, f"{prefix}{dbm}{suffix}")
        samples.append(load_csv_matrix(path, skiprows=skiprows))
    return np.array(samples)


def _normalize_dbm_label(dbm: str | int) -> str:
    label = str(dbm)
    if not label.endswith('dbm'):
        label = f'{label}dbm'
    return label


def load_dual_polarization_dataset(
    directory: str,
    real_prefix: str = 'DP_RealConstellationDiagram_',
    ideal_filename: Optional[str] = None,
    dbm_values: Optional[List[int]] = None,
    suffix: str = '.csv',
    skiprows: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Carrega dados reais e opcionais de constelação ideal.

    O conjunto real segue a convenção:
    DP_RealConstellationDiagram_{dbm}dbm[_{distance}].csv
    """
    if dbm_values is None:
        dbm_values = list(range(12))

    real_data = []
    missing_dbms = []
    for dbm in dbm_values:
        dbm_label = _normalize_dbm_label(dbm)
        primary_path = os.path.join(directory, f"{real_prefix}{dbm_label}{suffix}")
        fallback_path = os.path.join(directory, f"{real_prefix}{dbm}{suffix}")

        # Tenta o caminho com e sem 'dbm' no nome do arquivo.
        if os.path.isfile(primary_path):
            path = primary_path
        elif os.path.isfile(fallback_path):
            path = fallback_path
        else:
            missing_dbms.append(str(dbm))
            continue

        real_data.append(load_csv_matrix(path, skiprows=skiprows))

    if not real_data:
        raise FileNotFoundError(
            f'Nenhum arquivo de constelação real encontrado em {directory} com prefixo {real_prefix} ' 
            f'e valores dBm {dbm_values}.'
        )

    if missing_dbms:
        import warnings

        warnings.warn(
            f'Arquivos de constelação ausentes para valores dBm: {", ".join(missing_dbms)}. '
            'Foram carregados apenas os arquivos disponíveis.',
            UserWarning,
        )

    ideal_data = None
    if ideal_filename is not None:
        ideal_path = os.path.join(directory, ideal_filename)
        ideal_data = load_csv_matrix(ideal_path, skiprows=skiprows)

    return np.array(real_data), ideal_data


def split_series(data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Divide os dados de série temporal em treino e teste usando o último eixo."""
    if data.ndim < 2:
        raise ValueError('Esperado dados com pelo menos duas dimensões')

    n_samples = data.shape[-1]
    split_index = int(n_samples * train_ratio)

    # Usa fatiamento para separar as amostras de treino e de teste.
    train = data[..., :split_index]
    test = data[..., split_index:]
    return train, test


def plot_constellation(
    x: np.ndarray,
    y: np.ndarray,
    title: Optional[str] = None,
    bins: int = 80,
    figsize: Tuple[int, int] = (6, 6),
) -> None:
    """Mostra o diagrama de constelação em um histograma 2D."""
    plt.figure(figsize=figsize)
    plt.hist2d(x, y, bins=bins, cmap='viridis')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.title(title or 'Constellation Diagram')
    plt.colorbar(label='Counts')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def find_constellation_files(directory: str, pattern: str = 'DP_RealConstellationDiagram_*') -> List[str]:
    """Retorna a lista de caminhos que batem com o padrão de arquivos de constelação."""
    paths = sorted(glob(os.path.join(directory, pattern)))
    return paths
