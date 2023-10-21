# ai-project

Using AI to predict whether new music will go viral on social media.

### Setup

1. Setup virtual env, install requirements, and setup jupyter kernel:

   ```
       python3 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt
       python -m ipykernel install --user --name=venv
   ```
2. Duplicate .env.example, rename to .env and replace keys (only if need to use spotify api, aka run the d) and path to src directory.

#### Running the Notebooks

### Data Loading

Here we are using raw data scraped from [chartex.com](https://chartex.com/tiktok-music-chart-top-songs-from-tiktok/sort/number-videos-desc), which we saved in `data/chartex`. Each JSON file contains data for 200 tracks, including their Spotify ids which is what interests us.
Because we are using the Spotify API to run this notebook, it is necessary to obtain an API key (for free) from [the Spotify developer website](https://developer.spotify.com/documentation/web-api/tutorials/getting-started). Next, duplicate .env.example, rename to .env and replace keys.
Now we can run the notebook. You will be prompted to log in to Spotify from your browser.
Note getting data from all the pages may take a while, like many other notebooks in this project.