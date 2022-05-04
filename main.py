import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt

signal, fs = sf.read("HolySteel.wav", always_2d=True)

h = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [0.1, 2.5, 1.8],
        [0.1, 5, 1.8],
        [3, 5, 1.8],
    ],
    s=[3, 2.5, 1.8],          # Source position [x y z] (m)
    L=[6, 5, 3.5],            # Room dimensions [x y z] (m)
    reverberation_time=0.38, # Reverberation time (s)
    nsample=4096,           # Number of output samples
)

print(h.shape)              # (4096, 3)
print(signal.shape)         # (11462, 2)

# Convolve 2-channel signal with 3 impulse responses
signal = ss.convolve(h[:, None, :], signal[:, :, None])

print(signal.shape)         # (15557, 2, 3)

plt.plot