import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

rt60 = 0.38  # Seconds

#       Room Dimensions         #
roomYogaDim = [6, 5, 3.5]
roomShopDim = [5.5, 4, 3.5]
roomReceptionDim = [5.5, 4, 3.5]
roomSaunaDim = [6, 4, 3.5]
roomGymDim = [6, 10, 3.5]
roomSwimmingPoolDim = [10, 8, 5]
roomJacuzziDim = [6, 8, 3.5]
roomToiletsDim = [6, 3, 3]
roomCafeDim = [12, 4, 3.5]
roomLockerDim = [6, 5, 3]
roomBarDim = [12, 6, 3.5]
micLoc = np.c_[
    [3, 2.5, 3]
]

audio = wavfile.read('HolySteel.wav')

e_absorption, maxOrder = pra.inverse_sabine(rt60, roomBarDim)
roomBar = pra.ShoeBox(roomBarDim, fs=16000, materials=pra.Material(e_absorption))
roomBar.add_source([3, 2.5, 1.8], signal=audio, delay=1.3)
roomBar.add_microphone_array(micLoc)

roomBar.mic_array.to_wav(
    f"HolySteel.wav",
    norm=True,
    bitdepth=np.int16,
)

roomBar.compute_rir()

import matplotlib.pyplot as plt

plt.plot(roomBar.rir[1][0])
plt.show()
