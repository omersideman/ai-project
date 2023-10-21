from pytube import YouTube
import os
import librosa
import numpy as np
from youtube_search import YoutubeSearch
import signal
from pychorus import find_and_output_chorus


def dl_and_extract_features(track_info, output_directory='../data/audio_wav', delete_track=False):

    audio_path = download_track(track_info, output_directory)

    if audio_path is None:
        return None

    features = extract_features(audio_path)

    if delete_track:
        os.remove(audio_path)

    return features


def download_from_youtube(url, output_directory, filename):
    print(f"Downloading track {filename}")
    yt = YouTube(url)
    try:
        yt.streams.filter(only_audio=True).first().download(  # type: ignore
            output_path=output_directory, filename=filename)
    except Exception as e:
        print(f'Could not download {filename}. Error: {e}')
        return None

    audio_path = os.path.join(output_directory, filename)
    print(f"Downloaded audio to {audio_path}")
    return audio_path


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True, duration=60)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    feature_names = ["chroma_stft", "rmse", "spec_cent",
                     "spec_bw", "rolloff", "zcr", "mfcc"]
    feature_values = [chroma_stft, rmse,
                      spec_cent, spec_bw, rolloff, zcr, mfcc]

    # print feature names and size of each feature
    for feat, name in zip(feature_values, feature_names):
        print(f"{name}: {feat.shape}")

    means = [np.mean(feat) for feat in feature_values]

    features = dict(zip(feature_names, means))

    print(f"Extracted features: {features}")

    return features


def find_youtube_url(track_info):
    artist = track_info["artist"]
    title = track_info["track_name"]
    query = artist + " " + title + " lyrics"
    result = YoutubeSearch(query, max_results=1).to_dict()[0]
    # print(result)
    url = 'https://www.youtube.com' + result['url_suffix']  # type: ignore
    print(f'Found youtube url: {url}')
    return url


def download_track(track_info, output_directory):
    youtube_url = find_youtube_url(track_info)
    if not youtube_url:
        return None
    # filename = track_info["artist"] + " - " + track_info["track_name"] + ".wav"
    filename = track_info["id"] + ".wav"
    os.makedirs(output_directory, exist_ok=True)
    audio_path = download_from_youtube(youtube_url, output_directory, filename)
    return audio_path


def load_audio_with_timeout(audio_path, offset, duration, sample_rate=22050, timeout=5):
    def handler(signum, frame):
        raise TimeoutError('Timeout loading audio file')
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        y, sr = librosa.core.load(
            audio_path, sr=sample_rate, offset=offset, duration=duration)
        yt, _ = librosa.effects.trim(y)

    finally:
        signal.alarm(0)
    return yt, sr


def find_chorus(audio_path, duration, output_file=None):
    chorus_start_sec = find_and_output_chorus(
        input_file=audio_path, output_file=output_file, clip_length=duration)
    return chorus_start_sec


def find_chorus_with_timeout(audio_path, duration, timeout=10, output_file=None):
    def handler(signum, frame):
        raise TimeoutError('Timeout loading audio file')
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        chorus_start_sec = find_chorus(audio_path, duration, output_file)
    finally:
        signal.alarm(0)
    return chorus_start_sec
