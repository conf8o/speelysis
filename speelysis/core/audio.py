import numpy as np
import matplotlib.pyplot as plt
from typing import Generator

from .signal import fir
from .math import time_axis


class Audio:
    """音声分析用クラス
    
    Attributes:
        rate (int): サンプリング周波数
        data (np.ndarray[shape=(n,)]): 1次元のデータ
        times (np.ndarray[shape=data.shape, dtype=float]): 時間軸
    """
    
    def __init__(self, rate: int, data: np.ndarray):
        self.data = data.astype(np.float64)
        self.rate = rate
        self.times = time_axis(self.data, self.rate)

    def plot(self) -> None:
        """横軸を時間軸、縦軸をオーディオデータとしたグラフをプロットする。"""
        
        plt.plot(self.times, self.data)
        
    def each_frame(self, n_frame: int, step_ms: int) -> Generator[np.ndarray, None, None]:
        """オーディオデータを指定したフレーム長で切り取っていくジェネレータを取得する。
        
        Args:
            n_frame (int): フレーム長
            step_ms (int): ずらす長さ(ミリ秒)

        Yields:
            np.ndarray[shape=(n_frame,)]: オーディオデータをフレーム長で切り取ったデータ。
            step_ms(ミリ秒)ずつずらしながら切り取っていく。
            余った部分は取得しない。
        """
        
        step = self.rate * step_ms // 1000
        
        i = 0
        n = len(self.data)
        while i+n_frame <= n:
            yield self.data[i:i+n_frame]
            i += step

    def high_pass_filtered(self) -> 'Audio':
        """高域強調したAudioクラスのインスタンスを取得する。
        
        Returns:
            Audio: 高域強調後のAudio
        """

        return Audio(self.rate, fir(self.data))
