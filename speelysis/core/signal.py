import numpy as np


def fir(data: np.ndarray, a=0.97) -> np.ndarray:
    """有限インパルス応答

    Args:
        data (numpy.ndarray[int16]): 任意のデータ
        a (numpy.float64): 係数

    Returns:
        numpy.ndarray[numpy.float64]: フィルタ後データ
    """

    n = len(data)
    y = np.empty(n)
    y[0] = data[0]
    for i in range(1, n):
        y[i] = data[i] - a * data[i-1]

    return y