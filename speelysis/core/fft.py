import numpy as np


def fft(w: complex, s: np.ndarray) -> np.ndarray:
    """高速フーリエ変換(Fast Fourier Transform, FFT)

    w はオイラーの公式から求められる複素数 exp(-i2π/n)
    F(t) = w^t としたとき、F(0) = 1, F(n) = exp(-i2π/n)^n = exp(-i2π) = 1 となる。
    つまり周期はnである。関数Fは、イメージとしては複素平面上で時計回りに単位円を描く複素数を出力する周期nの関数である。 

    FFTは以下の条件を求める。
    (A) w^n = 1
    (B) n が2のべき乗である

    Args:
        w (complex): 複素数 exp(-i2π/n)
        s (np.ndarray[shape=(2**p,)]): データ

    Returns:
        np.ndarray[shape=(2**p,), dtype=complex]: フーリエ変換後の値
    """
    
    n = len(s)
    if n == 1: return s

    # 高速化が無駄になるため、チェックはコメントアウト
    # 実部を有効桁数10桁で丸めて1と比較。
    # if np.round(np.real(w ** n), 10) == 1 or n & (n - 1) == 0:
    #     raise FFTError("FFT Invalid Condition")

    f0 = fft(w*w, s[::2])
    f1 = fft(w*w, s[1::2])

    return np.concatenate([[f0[j] + w**j*f1[j] for j in range(n//2)],
                           [f0[j] + w**(n/2+j)*f1[j] for j in range(n//2)]])


class FFTError(Exception):
    """FFTの条件についてのエラー"""

    pass
