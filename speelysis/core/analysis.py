import numpy as np
import matplotlib.pyplot as plt
from typing import Generator


def time_axis(data: np.ndarray, rate: np.float64) -> np.ndarray:
    """NumPy配列とサンプリング周波数から時間軸を取得する。

    Args:
        data (numpy.ndarray[int16]): 任意のデータ
        rate (numpy.float64): サンプリング周波数

    Returns:
        numpy.ndarray[numpy.float64]: aryと同じ要素数を持つ時間軸
    """
    
    return np.arange(len(data)) / rate


class Audio:
    """音声分析用クラス
    
    Attributes:
        rate (int): サンプリング周波数
        data (numpy.ndarray[numpy.int16]): 1次元のデータ
        times (numpy.ndarray[numpy.int16]): 時間軸
    """
    
    def __init__(self, rate: int, data: np.ndarray):
        self.data = data
        self.rate = rate
        self.times = time_axis(self.data, self.rate)

    def plot(self) -> None:
        """横軸を時間軸、縦軸をオーディオデータとしたグラフをプロットする。"""
        
        plt.plot(self.times, self.data)

        return
        
    def each_frame(self, n_frame: int, step_ms: int) -> Generator[np.ndarray, None, None]:
        """オーディオデータを指定したフレーム長でずらしながら切り取っていくジェネレータを取得する。
        
        Args:
            n_frame (int): フレーム長
            step_ms (int): ずらす長さ(ミリ秒)

        Yields:
            numpy.ndarray[numpy.int16]: オーディオデータをフレーム長で切り取ったデータ。
            step_ms(ミリ秒)ずつずらしながら切り取っていく。
            余った部分は取得しない。
        """
        
        step = self.rate * step_ms // 1000
        
        i = 0
        n = len(self.data)
        while i+n_frame <= n:
            yield self.data[i:i+n_frame]
            i += step

    def high_path_filtered(self) -> 'Audio':
        """高域強調した本クラスのインスタンスを取得する
        
        Returns:
            Audio: 高域強調されたAudio
        """

        return Audio(self.rate, high_path_filter(self.data))


def high_path_filter(data: np.ndarray, a = 0.97) -> np.ndarray:
    """高域強調のための離散ハイパスフィルタ

    Args:
        data (numpy.ndarray[int16]): 任意のデータ
        a (numpy.float64): 係数

    Returns:
        numpy.ndarray[numpy.float64]: 高域強調
    """

    n = len(data)
    y = np.empty(n)
    y[0] = data[0]
    for i in range(1, n):
        y[i] = data[i] - a * data[i-1]

    return y


def frame_candidates(rate: int, min_ms: int, max_ms: int) -> Generator[int, None, None]:
    """サンプリング周波数から、窓関数で切り取るフレーム長の候補のジェネレータを取得する。

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


def stft(a: Audio, window: np.ndarray, step_length: int) -> Generator[np.ndarray, None, None]:
    """短時間フーリエ変換
    
    step_length(ms)ごとに、オーディオデータ(a)をフレーム長(frame_length)の範囲で切り取っていき、
    それぞれに窓関数(window)を適用し、高速フーリエ変換する。

    Args:
        a (Audio): オーディオ
        window (numpy.ndarray[numpy.float64]): 窓関数
        step_length (int): ずらす長さ(ミリ秒)
    """
    
    for frame in a.each_frame(len(window), step_length):
        windowed: np.ndarray = frame * window
        ffted: np.ndarray = np.fft.rfft(windowed)
        
        yield ffted
