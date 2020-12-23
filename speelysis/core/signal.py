import numpy as np


def fir(data: np.ndarray, a=0.97) -> np.ndarray:
    """有限インパルス応答

    Args:
        data (np.ndarray[shape=(n,)]): 任意のデータ
        a (float): 係数

    Returns:
        np.ndarray[shape=(n,), dtype=float]: フィルタ後データ
    """

    n = len(data)
    y = np.empty(n)
    y[0] = data[0]
    for i in range(1, n):
        y[i] = data[i] - a * data[i-1]

    return y