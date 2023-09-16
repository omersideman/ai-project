import librosa.display
import numpy as np
import pandas as pd
import os
import librosa
import librosa
import numpy
import skimage.io
from src.utils.file_utils import create_dirs_if_not_exist
from src.utils.audio_utils import load_audio_with_timeout, find_chorus_with_timeout


SR = 22050


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def save_spectrogram(spec, out_path):
    create_dirs_if_not_exist(out_path)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(spec, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255-img  # invert. make black==more energy
    # print(f"Image shape: {img.shape}")
    # save as PNG
    skimage.io.imsave(out_path, img)


def get_spectrograms(audio_directory, input_csv, output_directory, duration, start_index, end_index=0, res='low', chorus=False) -> list[np.ndarray]:
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

    # set resolution of spectrogram - bigger hops for lower resolution
    hop_length = 2048 if res == 'low' else 1024

    if end_index == 0:
        end_index = len(spotify_df)

    # Looping through each row in the df
    for i in range(start_index, end_index):

        print(f'Processing track {i} of {len(spotify_df)}')

        track_data = spotify_df.iloc[i]

        # Loading in the audio file
        spotify_id = track_data['id']
        audio_path = os.path.join(audio_directory, f'{spotify_id}.wav')
        image_path = os.path.join(output_directory, f'{spotify_id}.png')

        if os.path.exists(image_path):
            print(f"specogram already exists: {image_path}, skipping...")
            continue

        # getting the chorus start time
        if chorus:
            try:
                offset = find_chorus_with_timeout(audio_path, duration) or 0
            except Exception as e:
                print(
                    f"WARNING: problem with finding chorus: {e}, skipping...")
                offset = 0
        else:
            offset = 0

        # Computing the mel spectrogram
        try:
            spec = get_mel_spectrogram(
                audio_path, offset, duration, hop_length)
        except FileNotFoundError:
            print(f"WARNING: file not found: {audio_path}, skipping...")
            continue
        except TimeoutError:
            print(
                f"WARNING: file loading timed out for {audio_path}, skipping...")
            continue

        # adding to the list
        mel_specs.append(spec)

        # saving as png
        save_spectrogram(spec, image_path)

    return mel_specs


def get_mel_spectrogram(audio_path, offset, duration, hop_length) -> np.ndarray:
    """ This function takes in an audio file path, computes the
    mel spectrogram for the audio file, reshapes it so that it is the
    same size as the other spectrograms, and returns it as a numpy array."""

    # check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"File not found: {audio_path}")  # type: ignore

    # load audio with time out of five seconds
    try:
        y, sr = load_audio_with_timeout(
            audio_path, offset, duration)
    except:
        raise TimeoutError(f'Loading file: {audio_path} timed out')

    # settings
    n_mels = 128  # number of bins in spectrogram. Height of image

    # Computing the mel spectrograms
    spec = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=n_mels)
    spec = librosa.power_to_db(spec, ref=np.max)  # converting to decibals

    return spec
