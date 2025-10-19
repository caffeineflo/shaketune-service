import typing as t
from functools import partial

import numpy as np
import numpy.typing as npt


def _detrend_constant(data: npt.NDArray[np.float64]) -> npt.NDArray[np.floating[t.Any]]:
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'

    return data - np.mean(data, axis=-1, keepdims=True)


def _fft_helper(
    x: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
    nperseg: int,
    noverlap: int,
    nfft: int,
    sides: str,
) -> npt.NDArray[np.complex128]:
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Detrend each data segment individually
    result = _detrend_constant(result)

    # Apply window by multiplication
    result = window * result

    if sides == 'twosided':
        result = np.fft.fft(result, n=nfft)
    else:
        result = np.fft.rfft(result.real, n=nfft)

    return result


def spectrogram(x: npt.ArrayLike, fs: float, window: npt.NDArray[np.floating[t.Any]], nperseg: int, noverlap: int):
    axis: int = -1
    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    outdtype = np.result_type(x, np.complex64)

    # Early return
    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    if np.result_type(window, np.complex64) != outdtype:
        window = window.astype(outdtype)

    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    nfft = nperseg
    scale = 1.0 / (fs * (window * window).sum())

    if np.iscomplexobj(x):
        sides = 'twosided'
        freqs = np.fft.fftfreq(nfft, 1 / fs)
    else:
        sides = 'onesided'
        freqs = np.fft.rfftfreq(nfft, 1 / fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, window, nperseg, noverlap, nfft, sides)

    # PSD
    result = np.conjugate(result) * result
    result *= scale

    if sides == 'onesided':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    result = result.real

    # Roll frequency axis back to axis where the data came from
    result = np.moveaxis(result, -1, axis - 1)

    return freqs, time, result


# This is Klipper's spectrogram generation function adapted to use Scipy
def compute_spectrogram(
    data: npt.NDArray[np.floating[t.Any]],
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Sampling frequency
    fs = data.shape[0] / (data[-1, 0] - data[0, 0])
    # Round up to a power of 2 for faster FFT
    nperseg = 1 << int(0.5 * fs - 1).bit_length()
    # Calculate Kaiser window
    window = np.kaiser(nperseg, 6.0)

    # Spectrogram helper
    _specgram = partial(spectrogram, fs=fs, window=window, nperseg=nperseg, noverlap=nperseg // 2)

    f, t, pdata = _specgram(data[:, 1])
    for axis in range(2):
        pdata += _specgram(data[:, axis + 2])[2]

    return pdata.astype(np.float64), t, f
