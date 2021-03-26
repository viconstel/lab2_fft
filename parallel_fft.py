import numpy as np
import time
import multiprocessing as mp
from scipy.fft import fft


class MyParallelFFT:

    @staticmethod
    def dft(x):
        X_fourier = np.zeros_like(x, dtype=complex)
        N = len(x)

        for k in range(len(X_fourier)):
            values = np.array([x_n * np.exp(-2j * np.pi * k * n / N) for n, x_n in enumerate(x)])
            X_fourier[k] = np.sum(values)

        return X_fourier

    def custom_fft(self, x):
        N = len(x)

        if N % 2 > 0:
            return self.dft(x)
        else:
            X_even = self.custom_fft(x[::2])
            X_odd = self.custom_fft(x[1::2])
            return self.merge_transformed_part(X_even, X_odd, N)

    def run_fft(self, x):
        N = len(x)
        with mp.Pool(2) as p:
            res = p.map(self.custom_fft, [x[::2], x[1::2]])
        return self.merge_transformed_part(res[0], res[1], N)

    @staticmethod
    def merge_transformed_part(x, y, N):
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([x + factor[:N // 2] * y,
                               x + factor[N // 2:] * y])

    @staticmethod
    def create_signal(n, duration=1000):
        x = np.linspace(0, duration, n)
        amp, freq, phase = np.random.randint(0, 50, 3)
        sin_harmonic = amp * np.sin(freq * np.pi * x) + phase
        amp, freq, phase = np.random.randint(0, 50, 3)
        cos_harmonic = amp * np.cos(freq * np.pi * x) + phase
        signal = sin_harmonic + 0.5 * cos_harmonic
        return signal


if __name__ == '__main__':
    signal_sizes = [2 ** 9, 2 ** 12, 2 ** 15, 2 ** 18, 2 ** 21]
    my_fft = MyParallelFFT()

    for i, size in enumerate(signal_sizes):
        start = time.time()
        sig = my_fft.create_signal(size)
        sig_fourier = my_fft.run_fft(sig)
        sig_scipy = fft(sig)
        print(f'Сигнал: {i+1}, проверка: {np.allclose(sig_scipy, sig_fourier)}, '
              f'время: {time.time() - start}')
