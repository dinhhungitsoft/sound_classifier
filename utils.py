import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def show_spectrogram(melspectrum, sample_rate, out_file=None):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(melspectrum.squeeze(0).numpy(), ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sample_rate,
                            ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    if out_file:
        plt.savefig(out_file)
    plt.show()