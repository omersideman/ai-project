from pytube import YouTube
import os
import librosa
import numpy as np
from youtube_search import YoutubeSearch


def download_from_youtube(url, output_directory, filename):
    yt = YouTube(url)
    yt.streams.filter(only_audio=True).first().download(  # type: ignore
        output_path=output_directory, filename=filename)
    return os.path.join(output_directory, yt.title + ".mp4")


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features = [chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc]

    for feat in features:
        print(feat.shape)

    means = [np.mean(feat) for feat in features]
    return means


def find_youtube_url(track_info):
    artist = track_info["artist"]
    title = track_info["title"]
    query = artist + " " + title + " lyrics"
    result = YoutubeSearch(query, max_results=1).to_dict()[0]
    url = 'https://www.youtube.com' + result['link']  # type: ignore
    print(url)
    return url
