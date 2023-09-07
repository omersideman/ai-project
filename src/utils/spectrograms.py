import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm
from PIL import Image
from src.utils.file_utils import create_dirs_if_not_exist
from src.utils.audio_utils import load_audio_with_timeout


def get_spectrograms(audio_directory, input_csv, output_directory, start_index, end_index) -> list[np.ndarray]:
    """
    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for the files corresponding to the rows start_index to end_index in the input_csv,
    reshapes them so that they are all the same size, and saves them as images to output directory.
    Returns a list of the mel spectrograms as numpy arrays."""

    create_dirs_if_not_exist(output_directory)

    # loading dataframe
    spotify_df = pd.read_csv(input_csv)

    # Creating empty lists for mel spectrograms and labels
    mel_specs = []

    # Setting the size for all mel spectrograms
    spectrogram_size = (128, 1292)

    # Looping through each row in the df
    for i in range(start_index, end_index):

        print(f'Processing row {i} of {len(spotify_df)}')

        track_data = spotify_df.iloc[i]

        # Loading in the audio file
        spotify_id = track_data['id']
        audio_path = os.path.join(audio_directory, f'{spotify_id}.wav')
        image_path = os.path.join(output_directory, f'{spotify_id}.jpg')

        if os.path.exists(image_path):
            print(f"Spectogram already exists: {image_path}, skipping...")
            continue

        # Computing the mel spectrogram
        try:
            spect = get_mel_spectrogram(audio_path, spectrogram_size)
        except FileNotFoundError:
            print(f"WARNING: file not found: {audio_path}, skipping...")
            continue
        except TimeoutError:
            print(
                f"WARNING: file loading timed out for {audio_path}, skipping...")
            continue

        # Saving the mel spectrogram as an image
        print(f'SHape: {spect.shape}')
        im = Image.fromarray(spect)
        im.save(image_path)

        # adding to the list
        mel_specs.append(spect)

    return mel_specs


def get_mel_spectrogram(audio_path, size=(128, 1292)) -> np.ndarray:
    """ This function takes in an audio file path, computes the
    mel spectrogram for the audio file, reshapes it so that it is the
    same size as the other spectrograms, and returns it as a numpy array."""

    # check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}") # type: ignore

    # load audio with time out of five seconds
    try:
        y, sr = load_audio_with_timeout(audio_path)
    except:
        raise TimeoutError(f'Loading file: {audio_path} timed out')

    # Computing the mel spectrograms
    spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)

    # Adjusting the size
    if spect.shape != size:
        spect.resize(*size, refcheck=False)

    return spect
