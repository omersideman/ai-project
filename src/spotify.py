
# Wrapper for Spotify API - https://spotipy.readthedocs.io/en/latest/#

import json
from os import environ
import time
from dotenv import load_dotenv
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from src.utils import pretty_print
from utils import create_dirs_if_not_exist
from requests.exceptions import ReadTimeout  # type: ignore

load_dotenv()

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


class Spotify:

    def __init__(self):
        pass

    def _call_spotify_api(self, func, *args, **kwargs):
        '''wrapper for spotify api calls to handle timeouts'''
        for i in range(3):
            try:
                return func(*args, **kwargs)
            except ReadTimeout:
                print('ReadTimeout - waiting 10 seconds')
                time.sleep(10)
                return func(*args, **kwargs)

    def get_playlist_info(self, playlist_url: str):
        ''' returns basic info about a playlist'''

        # should use fields param to make less api calls but not working
        info = self._call_spotify_api(sp.playlist, playlist_url)

        if not info:
            raise ValueError('No results found')

        res = {'name': info['name'], 'id': info['id'],  # type: ignore
               'total': info['tracks']['total']}  # type: ignore
        return res

    def get_playlist_tracks(self, playlist_url: str):
        '''returns all tracks in a playlist'''
        results = self._call_spotify_api(sp.playlist_items, playlist_url)

        if not results:
            raise ValueError('No results found')

        tracks = results['items']

        while results['next']:  # type: ignore
            results = sp.next(results)
            tracks.extend(results['items'])  # type: ignore

        print(f'{len(tracks)} tracks in playlist')

        return tracks

    def playlist_to_df(self, playlist_url: str,):
        '''returns a dataframe of all tracks in a playlist with only relevant info'''

        print(f'Getting playlist tracks from spotify for {playlist_url}')

        results = self._call_spotify_api(
            self.get_playlist_tracks, playlist_url)

        if not results:
            raise ValueError('No results found')

        tracks = [t['track'] for t in results]

        clean = []

        print('Creating dataframe')

        for i, track in enumerate(tracks):
            # print(f'Track {i}')
            artist_uri = track['artists'][0]['uri']
            # artist_info = self._call_spotify_api(sp.artist, artist_uri)
            audio_features = self._call_spotify_api(
                sp.audio_features, track['uri'])[0]  # type: ignore
            print(f'Popularity: {track["popularity"]}')
            record = {'track_name': track['name'],
                      'track_pop': track['popularity'],
                      'artist': track['artists'][0]['name'],
                      #   'artist_pop': artist_info['popularity'],  # type: ignore
                      'album': track['album']['name'],
                      'length': track['duration_ms'],
                      'track_uri': track['uri']}

            record.update(audio_features)
            clean.append(record)

        df = pd.json_normalize(clean)

        return df

    def combine_to_csv(self, playlist_urls: list, name: str):
        '''end to end function to get tracks from multiple playlists and save to csv'''

        dataframes = [self.playlist_to_df(url) for url in playlist_urls]
        combined_df = pd.concat(dataframes, ignore_index=True)
        unique_df = combined_df.drop_duplicates()

        print(f'{len(combined_df)} tracks in total')

        filepath = f'../data/{name}.csv'

        create_dirs_if_not_exist(filepath)

        unique_df.to_csv(filepath, index=False)

        return unique_df
