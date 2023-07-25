

from os import environ
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


def test_spotipy():
    playlists = sp.user_playlists('spotify')
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            print("%4d %s %s" %
                  (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None
