import numpy as np
from typing import Generator
import itertools

from .audio import Audio


def frame_candidates(rate: int, min_ms: int, max_ms: int) -> Generator[int, None, None]:
    """サンプリング周波数をもとに窓関数のフレーム長候補のジェネレータを取得する。

    フレーム長は2のべき乗になる

    Args:
        rate (int): サンプリング周波数

    Yields:
        int: フレーム長(2のべき乗)
    """

    min_n = rate * min_ms / 1000
    max_n = rate * max_ms / 1000

    for p in itertools.count(1):
        n = 1 << p
        if min_n < n < max_n:
            yield n
        elif n > max_n:
            break


def stft(a: Audio, window: np.ndarray, step_length: int) -> Generator[np.ndarray, None, None]:
    """短時間フーリエ変換
    
    step_length(ms)ごとにオーディオデータを窓関数で切り取っていき、それぞれ高速フーリエ変換する。

    Args:
        a (Audio): オーディオ
        window (numpy.ndarray[numpy.float64]): 窓関数
        step_length (int): ずらす長さ(ミリ秒)

    Yields:
        numpy.ndarray[numpy.complex128]: 実FFT適用後のデータ
    """
    
    for frame in a.each_frame(len(window), step_length):
        windowed: np.ndarray = frame * window
        ffted: np.ndarray = np.fft.rfft(windowed)
        
        yield ffted
