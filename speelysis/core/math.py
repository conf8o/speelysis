import numpy as np


def time_axis(data: np.ndarray, rate: int) -> np.ndarray:
    """NumPy配列とサンプリング周波数から時間軸を取得する。

    Args:
        data (numpy.ndarray[int16]): 任意のデータ
        rate (int): サンプリング周波数

    Returns:
        numpy.ndarray[numpy.float64]: aryと同じ要素数を持つ時間軸
    """
    
    return np.arange(len(data)) / rate


def sin_wave(k: int, rate: int, ms: int) -> np.ndarray:
    """サンプリング周波数rate(Hz)におけるms(ミリ秒)までの周期kのsin波を取得する。

    Args:
        k (int): 周波数
        rate (int): サンプリング周波数
        ms (int): ミリ秒
    
    Returns:
        numpy.ndarray[numpy.float64]: sin波
    """
    
    xs = np.linspace(0, ms / 1000, rate * ms // 1000)
    w = 2 * np.pi * k
    return np.sin(xs * w)
