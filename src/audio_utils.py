from pytube import YouTube
import os
import librosa
import numpy as np
from youtube_search import YoutubeSearch
import tempfile


def get_audio_features(track_info, output_directory=None):
    youtube_url = find_youtube_url(track_info)
    if not youtube_url:
        return None
    filename = track_info["artist"] + " - " + track_info["track_name"] + ".wav"
    out_dir = output_directory or 'data/track_downloads'
    os.makedirs(out_dir, exist_ok=True)
    audio_path = download_from_youtube(youtube_url, out_dir, filename)
    features = extract_features(audio_path)

    if output_directory is None:
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
    y, sr = librosa.load(audio_path, mono=True, duration=30)
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
