import numpy as np


def time_axis(data: np.ndarray, rate: int) -> np.ndarray:
    """配列とサンプリング周波数から時間軸を取得する。

    Args:
        data (np.ndarray[shape=(n,)]): 任意のデータ
        rate (int): サンプリング周波数

    Returns:
        np.ndarray[shape=data.shape, dtype=float]: 時間軸
    """
    
    return np.arange(len(data)) / rate


def sin_wave(k: int, rate: int, ms: int) -> np.ndarray:
    """サンプリング周波数rate(Hz)におけるms(ミリ秒)までの周波数kのsin波を取得する。

    Args:
        k (int): 周波数
        rate (int >= 1000): サンプリング周波数
        ms (int): ミリ秒
    
    Returns:
        np.ndarray[shape=(rate * ms / 1000, ), dtype=float]: sin波
    """
    
    xs = np.linspace(0, ms / 1000, rate * ms // 1000)
    w = 2 * np.pi * k
    return np.sin(xs * w)
