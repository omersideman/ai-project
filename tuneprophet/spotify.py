
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

    def playlist_to_csv(self, playlist_url: str, name: str):

        tracks = self.get_playlist_tracks(playlist_url)

        if not tracks:
            raise ValueError('No results found')

        df = pd.json_normalize(tracks)
        filepath = f'../data/{name}.csv'
        create_dirs_if_not_exist(filepath)
        df.to_csv(filepath, index=False)

    def combine_to_csv(self, playlist_urls: list, name: str):
        track_lists = [self.get_playlist_tracks(
            url) for url in playlist_urls]
        combined_tracks = [
            track for tracklist in track_lists for track in tracklist]

        combined_tracks = [track['track'] for track in combined_tracks]

        tracks = []
        for track in combined_tracks:
            if track not in tracks:
                tracks.append(track)

        print(combined_tracks[0])

        # # combined_tracks = []
        # # for tracklist in track_lists:
        # #     combined_tracks.extend(tracklist)

        # unique_ids = set()
        # unique_tracks_list = []

        # for d in combined_tracks:
        #     pretty_print(d)
        #     if d['track']["id"] not in unique_ids:
        #         unique_ids.add(d['track']["id"])
        #     unique_tracks_list.append(d)

        # unique_tracks = set(combined_tracks)
        # unique_tracks_list = list(unique_tracks)

        print(len(combined_tracks))
        df = pd.DataFrame(combined_tracks)
        unique_df = df.drop_duplicates(subset="id")
        filepath = f'../data/{name}.csv'
        create_dirs_if_not_exist(filepath)
        unique_df.to_csv(filepath, index=False)
        return (filepath)
