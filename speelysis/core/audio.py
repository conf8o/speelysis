import numpy as np
import matplotlib.pyplot as plt
from typing import Generator

from .sound import high_path_filter
from .math import time_axis


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
        """高域強調したAudioクラスのインスタンスを取得する
        
        Returns:
            Audio: 高域強調後のAudio
        """

        return Audio(self.rate, high_path_filter(self.data))

