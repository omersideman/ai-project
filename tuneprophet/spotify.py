
# Wrapper for Spotify API - https://spotipy.readthedocs.io/en/latest/#

import json
from os import environ
from dotenv import load_dotenv
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from tuneprophet.utils import pretty_print
from utils import create_dirs_if_not_exist

load_dotenv()

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


class Spotify:

    def __init__(self):
        pass

    def get_playlist_info(self, playlist_url: str):
        # this is stupid but it's not working otherwise
        info = sp.playlist(playlist_url)

        if not info:
            raise ValueError('No results found')

        res = {'name': info['name'], 'id': info['id'],  # type: ignore
               'total': info['tracks']['total']}  # type: ignore
        return res

    def get_playlist_tracks(self, playlist_url: str):
        results = sp.playlist_items(playlist_url)

        if not results:
            raise ValueError('No results found')

        tracks = results['items']

        while results['next']:  # type: ignore
            results = sp.next(results)
            tracks.extend(results['items'])  # type: ignore
        print(len(tracks))
        return tracks

    def playlist_to_df(self, playlist_url: str,):

        results = self.get_playlist_tracks(playlist_url)

        if not results:
            raise ValueError('No results found')

        tracks = [t['track'] for t in results]

        clean = []

        for track in tracks:
            artist_uri = track['artists'][0]['uri']
            artist_info = sp.artist(artist_uri)
            audio_features = sp.audio_features(track['uri'])[0] # type: ignore

            record = {'track_name': track['name'],
                      'track_pop': track['popularity'],
                      'artist': track['artists'][0]['name'],
                      'artist_pop': artist_info['popularity'],  # type: ignore
                      'album': track['album']['name'],
                      'length': track['duration_ms'],
                      'track_uri': track['uri']}

            record.update(audio_features)
            clean.append(record)

        df = pd.json_normalize(clean)

        return df

    def combine_to_csv(self, playlist_urls: list, name: str):

        dataframes = [self.playlist_to_df(url) for url in playlist_urls]
        combined_df = pd.concat(dataframes, ignore_index=True)
        unique_df = combined_df.drop_duplicates()

        print(len(combined_df))

        filepath = f'../data/{name}.csv'

        create_dirs_if_not_exist(filepath)

        unique_df.to_csv(filepath, index=False)

        return unique_df
