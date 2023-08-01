import json
import requests


def url_to_json(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data and return it as a Python object
        return json.loads(response.content)
    else:
        # The request failed for some reason
        print(f"Error: {response.status_code} - {response.reason}")
        return None


def scrape_chartex():
    url = 'https://chartex.com/api/tiktok_songs/?pageSize=200&ordering=-number_videos&page=9'
    obj = url_to_json(url)
    print(obj)
