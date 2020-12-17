import numpy as np
from typing import Callable


def mel_scale(f0: np.float64) -> Callable[[np.float64], np.float64]:
    """メル尺度

    Args:
        f0 (numpy.float64): 自由パラメータの周波数

    Returns:
        Callable[[numpy.float64], numpy.float64]: 周波数を受け取りメル尺度を返す関数
    """

    m0 = 1000.0 / np.log10(1000.0 / f0 + 1.0)

    def mel_scaled(f: np.float64) -> np.float64:
        return m0 * np.log10(f / f0 + 1.0)

    return mel_scaled


def imel_scale(f0: np.float64) -> Callable[[np.float64], np.float64]:
    """メル尺度の逆関数
    
    Args:
        f0 (numpy.float64): 自由パラメータの周波数

    Returns:
        Callable[[numpy.float64], numpy.float64]: メル尺度を受け取り周波数を返す関数
    """
    
    m0 = 1000.0 / np.log10(1000.0 / f0 + 1.0)
    
    def imel_scaled(m: np.float64) -> np.float64:
        return f0 * (10 ** (m / m0) - 1)
    
    return imel_scaled


def spl(p: np.float64) -> np.float64:
    """音圧レベル

    Args:
        p (numpy.float64): 音圧

    Returns:
        numpy.float64: 音圧レベル
    """

    p0 = 20 * (10**6)
    return 20 * np.log10(p / p0)
