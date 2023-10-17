
# Wrapper for Spotify API - https://spotipy.readthedocs.io/en/latest/#

from os import environ
import time
from dotenv import load_dotenv
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from src.utils.file_utils import create_dirs_if_not_exist
from requests.exceptions import ReadTimeout  # type: ignore

load_dotenv()

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

scope = "playlist-modify-private playlist-modify-public"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))


class Spotify:

    def __init__(self):
        pass

    def _call_spotify_api(self, func, *args, **kwargs):
        '''wrapper for spotify api calls to handle timeouts'''

        max_attempts = 3
        backoff = 10

        for i in range(max_attempts):
            try:
                res = func(*args, **kwargs)
                if res == None:
                    raise ValueError('No results found')
                return res
            except ReadTimeout:
                print('ReadTimeout - waiting 10 seconds')
                time.sleep(backoff)

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
            # artist_uri = track['artists'][0]['uri']
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

    def get_track_info(self, track_uri: str):
        '''returns info about a track'''

        track = self._call_spotify_api(sp.track, track_uri)
        assert (track)

        artist = self._call_spotify_api(sp.artist, track['artists'][0]['uri'])
        assert (artist)

        basic_info = {'track_name': track['name'],
                      'track_pop': track['popularity'],
                      'artist': track['artists'][0]['name'],
                      'artist_pop': artist['popularity'],
                      'album': track['album']['name'],
                      'length': track['duration_ms'],
                      'track_uri': track['uri']}

        audio_features = self._call_spotify_api(
            sp.audio_features, track['uri'])[0]  # type: ignore

        if audio_features == None:
            print(
                f"WARNING: No Audio features found for track {self.id_to_url(track['id'])}")
            return basic_info

        basic_info.update(audio_features)

        return basic_info

    def create_spotify_playlist(self, track_ids, playlist_name):
        '''creates a spotify playlist with the given name and adds the tracks'''

        track_uris = [self.id_to_url(id) for id in track_ids]

        # get user id
        user = self._call_spotify_api(sp.current_user)
        user_id = user['id']  # type: ignore

        # create playlist
        playlist = self._call_spotify_api(
            sp.user_playlist_create, user_id, playlist_name, True)

        # spotify api only allows 100 tracks to be added at a time
        for i in range(0, len(track_uris), 100):
            print(i)
            self._call_spotify_api(sp.playlist_add_items,
                                   playlist['id'], track_uris[i:i + 100]) # type: ignore

        return playlist

    def id_to_url(self, track_id):
        '''returns a spotify url from a track id'''
        return f'https://open.spotify.com/track/{track_id}'
