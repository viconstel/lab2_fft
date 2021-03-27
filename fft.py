import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fft as scipy_fft
from parallel_fft import MyParallelFFT


class MyFFT:

    @staticmethod
    def dft(x):
        X_fourier = np.zeros_like(x, dtype=complex)
        N = len(x)

        for k in range(len(X_fourier)):
            values = np.array([x_n * np.exp(-2j * np.pi * k * n / N) for n, x_n in enumerate(x)])
            X_fourier[k] = np.sum(values)

        return X_fourier

    def run_fft(self, x):
        N = len(x)

        if N % 2 > 0:
            return self.dft(x)
        else:
            X_even = self.run_fft(x[::2])
            X_odd = self.run_fft(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(N) / N)
            return np.concatenate([X_even + factor[:N // 2] * X_odd,
                                   X_even + factor[N // 2:] * X_odd])

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
    signal_sizes = [2 ** 9, 2 ** 12, 2 ** 15, 2 ** 18, 2 ** 21, 2 ** 24]
    seq_times, parallel_times = [], []
    my_fft = MyFFT()

    print('Последовательная обработка:')
    for i, size in enumerate(signal_sizes):
        start = time.time()
        sig = my_fft.create_signal(size)
        sig_fourier = my_fft.run_fft(sig)
        sig_scipy = scipy_fft(sig)
        finish = time.time() - start
        seq_times.append(finish)
        print(f'Сигнал: {i + 1}, проверка: {np.allclose(sig_scipy, sig_fourier)}, '
              f'время: {finish}')

    print('Параллельная обработка:')
    parallel_fft = MyParallelFFT()
    for i, size in enumerate(signal_sizes):
        start = time.time()
        sig = parallel_fft.create_signal(size)
        sig_fourier = parallel_fft.run_fft(sig)
        sig_scipy = scipy_fft(sig)
        finish = time.time() - start
        parallel_times.append(finish)
        print(f'Сигнал: {i + 1}, проверка: {np.allclose(sig_scipy, sig_fourier)}, '
              f'время: {finish}')

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(signal_sizes, seq_times, 'o-', label='Последовательная обработка')
    ax1.plot(signal_sizes, parallel_times, '-o', label='Параллельная обработка')
    ax1.set_title('Время работы')
    ax1.set_xlabel('Размер сигнала')
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(212)
    acceleration = np.array(parallel_times) / np.array(seq_times)
    ax2.plot(signal_sizes, acceleration, 'o-')
    ax2.set_title('Среднее ускорение')
    ax2.set_xlabel('Размер сигнала')
    ax2.grid()

    fig.savefig('./total_stats.png')
