import numpy as np
from typing import Generator
import itertools

from .audio import Audio
from .sound import mel_scale, imel_scale


def frame_candidates(rate: int, min_ms: int, max_ms: int) -> Generator[int, None, None]:
    """サンプリング周波数をもとに窓関数のフレーム長候補のジェネレータを取得する。

    フレーム長は2のべき乗になる。

    Args:
        rate (int): サンプリング周波数
        min_ms (int): 欲しい最小のフレーム長(ミリ秒)
        max_ms (int): 欲しい最大のフレーム長(ミリ秒)

    Yields:
        int: フレーム長(要素数(2のべき乗))
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
    
    step_length(ミリ秒)ごとにオーディオデータを窓関数で切り取っていき、それぞれ高速フーリエ変換する。

    Args:
        a (Audio): オーディオ
        window (np.ndarray[shape=(2**p,)]): 窓関数
        step_length (int): ずらす長さ(ミリ秒)

    Yields:
        np.ndarray[shape=(1 + 2**p/2,), dtype=complex]: 実FFT適用後のデータ
    """
    
    for frame in a.each_frame(len(window), step_length):
        windowed: np.ndarray = frame * window
        ffted: np.ndarray = np.fft.rfft(windowed)
        
        yield ffted


def tri_window(fs: int, n: int, l: float, r: float) -> np.ndarray:
    """三角窓

    Args:
        fs (int): 最大周波数
        n (int): 要素数
        l (float): 三角窓 左下の周波数
        r (float): 三角窓 右下の周波数

    Returns:
        np.ndarray[shape=(n,), dtype=float]: 三角窓
    """

    l = l * n // fs
    r = r * n // fs

    edge = int((r - l + 1) / 2)
    
    a = np.zeros(int(l))
    b = np.linspace(0, 1, edge)
    c = np.linspace(0, 1, edge)[::-1]
    
    f = np.concatenate([a, b, c])
    return np.append(f, np.zeros(n - len(f)))
        

def mel_filter_bank(fs: int, n: int, n_bins: int, mel_param=700) -> np.ndarray:
    """メルフィルタバンク

    Args:
        f (int): 周波数
        n (int): 離散信号要素数
        n_bins (int): ビン数
        mel_param (int): メル尺度の自由パラメータ

    Returns:
        np.ndarray[shape=(n, n_bins), dtype=float]: メルフィルタバンク
    """

    mel_scaled = mel_scale(mel_param)
    imel_scaled = imel_scale(mel_param)

    mel_end = int(mel_scaled(fs)) + 1

    mel_list = np.arange(0, mel_end, mel_end // (n_bins + 1))

    hz_l_iter = iter(mel_list)
    hz_r_iter = iter(mel_list)
    next(hz_r_iter)
    next(hz_r_iter)

    hz_l_iter = (imel_scaled(b)
                 for b
                 in hz_l_iter)

    hz_r_iter = (imel_scaled(b)
                 for b
                 in hz_r_iter)
    
    return np.array([tri_window(fs, n, l, r) 
                     for l, r 
                     in zip(hz_l_iter, hz_r_iter)])


def mel_filter_bank_freq(fs: int, n_bins: int, mel_param=700) -> np.ndarray:
    """メルフィルタバンクの周波数軸

    Args:
        fs (int): 周波数最大値
        n_bins (int): ビン数

    Returns:
        np.ndarray[shape=(n_bins,), dtype=float]: メル尺度上で等間隔の周波数軸。三角窓の頂点の値を取る。
    """

    mel_scaled = mel_scale(mel_param)
    imel_scaled = imel_scale(mel_param)

    mel_end = mel_scaled(fs)
    return imel_scaled(np.arange(1, n_bins + 1) * (mel_end / (n_bins + 1)))
