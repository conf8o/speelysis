import numpy as np
from typing import Callable


def mel_scale(f0: float) -> Callable[[float], float]:
    """メル尺度

    Args:
        f0 (float): 自由パラメータの周波数

    Returns:
        Callable[[float], float]: 周波数を受け取りメル尺度を返す関数
    """

    m0 = 1000.0 / np.log10(1000.0 / f0 + 1.0)

    def mel_scaled(f: float) -> float:
        return m0 * np.log10(f / f0 + 1.0)

    return mel_scaled


def imel_scale(f0: float) -> Callable[[float], float]:
    """メル尺度の逆関数
    
    Args:
        f0 (float): 自由パラメータの周波数

    Returns:
        Callable[[float], float]: メル尺度を受け取り周波数を返す関数
    """
    
    m0 = 1000.0 / np.log10(1000.0 / f0 + 1.0)
    
    def imel_scaled(m: float) -> float:
        return f0 * (10 ** (m / m0) - 1)
    
    return imel_scaled


def spl(p: float) -> float:
    """音圧レベル

    Args:
        p (float): 音圧

    Returns:
        float: 音圧レベル
    """

    p0 = 20 * (10**6)
    return 20 * np.log10(p / p0)
