import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

# The desired reverberation time and dimensions of the room
rt60_tgt = (0.43513513513513513)  # seconds
room_dim = [6, 5, 4]  # meters

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read("Beep.wav")

# We invert Sabine's formula to obtain the parameters for the ISM simulator
#maxOrder = pra.inverse_sabine(rt60_tgt, room_dim)
m = pra.make_materials(
    ceiling="hard_surface",
    floor="ceramic_tiles",
    east="ceramic_tiles",
    west="ceramic_tiles",
    north="rough_concrete",
    south="rough_concrete",
)
# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=m, max_order=17
)

# place the source in the room
room.add_source([3, 2.5, 1.8], signal=audio, delay=0.5)

# define the locations of the microphones
mic_locs = np.c_[
    [3, 5, 1.8],
    [6, 2.5, 1.8],
    [6, 5, 1.8],
]

# finally place the array in the room
room.add_microphone_array(mic_locs)

# Run the simulation (this will also build the RIR automatically)
room.simulate()

room.mic_array.to_wav(
    f"BeepBar.wav",
    norm=True,
    bitdepth=np.int16,
)

# measure the reverberation time
rt60 = room.measure_rt60()
print("The desired RT60 was {}".format(rt60_tgt))
print("The measured RT60 is {}".format(rt60[1, 0]))

# Create a plot
plt.figure()

# plot one of the RIR. both can also be plotted using room.plot_rir()
rir_1_0 = room.rir[1][0]
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
plt.title("The RIR from source 0 to mic 1")
plt.xlabel("Time [s]")

# plot signal at microphone 1
plt.subplot(2, 1, 2)
plt.plot(room.mic_array.signals[1, :])
plt.title("Microphone 1 signal")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()