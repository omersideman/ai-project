# ai-project

Using AI to predict whether new music will go viral on social media.

## Enviroment setup

1. Setup virtual env, install requirements, and setup jupyter kernel:

   ```
       python3 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt
       python -m ipykernel install --user --name=venv
   ```

2. Duplicate .env.example, rename to .env and replace keys (only if need to use spotify api, aka run the d) and path to src directory.

## Submission directory structure and explanation

go-viral:
|- data: Containing all the data for this project.
    |-audio_mp3: All the .mp3 files of the songs would be saved here.
    |-audio_wav: All the .wav files of the songs would be saved here. We need it as well because when we download the audio files from youtube, we get them in .wav format.
    |-csv_files: Some intermediate .csv files of the dataset.
    |-specs: All the spectrogram of the songs would be saved here.
|- notebooks: containing all the jupyter notebooks we created in this project.
|- src:
    |-RNN_utils: We used the code here for both the RNN, CNN and Spotify feature's MLP.
        |-audio_utils.py: Some audio utils like rechanneling for the creation of the spectrogram.
        |-cross_val.py: The cross validation class and a function to print the cross validation results.
        |-dataset.py: The dataset classes that will be used for the dataloader.
        |-trainer.py: The trainining class for the deep models.
    |-utils: Some utils functions for different tasks we encouter thourgh the project.
        |-audio_utils.py
        |-csv_utils.py
        |-file_utils.py
        |-image_utils.py
        |-plot_utils.py
        |-spectrograms.py
        |-spotify
|-libs.txt
|-README.md
|-requirements.txt

## Notebook's explanations

#### `data_loading.ipynb`

Here we are using raw data scraped from [chartex.com](https://chartex.com/tiktok-music-chart-top-songs-from-tiktok/sort/number-videos-desc), which we saved in `data/chartex`. Each JSON file contains data for 200 tracks, including their Spotify ids which is what interests us.

Because we are using the Spotify API to run this notebook, it is necessary to obtain an API key (for free) from [the Spotify developer website](https://developer.spotify.com/documentation/web-api/tutorials/getting-started). Next, duplicate .env.example, rename to .env and replace keys.

Now we can run the notebook. You will be prompted to log in to Spotify from your browser.
Note getting data from all the pages may take a while, like many other notebooks in this project.

Thankfully, there is no real need to run this notebook because audio_features.csv is added to the submission and it is the result of this notebook.

#### `audio_features_explained.ipynb`

Notebook demonstrating the low-level audio features on a sample track, with explanations.

#### `audio_features.ipynbg`

Downloading the audio files and extracting the low-level audio features. Saving final dataset, to be used by ML models, to csv.

#### `data_analysis.ipynb`

Exploratory data analysis, including plotting feature distribution and examination of cross-correlations in our data.

#### `simple_classifyers.ipynb`

Initial attempt at classification using basic machine learning algorithms and models, such as SVM, KNN, Adaboost, decision trees, and MLP.

#### `spectrograms.ipynb`

Notebook explaining spectrograms. The creation of the spectrogram is in thr RNN.ipynb.

#### `RNN.ipynb`

Training and testing a recurrent neural network on audio files. Recommended to run with GPU. In addition, here we create the spectrogram for both the RNN testing and the CNN testing.

#### `CNN.ipynb`

Training and testing a convolutional neural network on audio spectrograms. Recommended to run with GPU.

#### `transformer.ipynb`

Using transformer for feature extraction from audio input. Required GPU with at least 15GB of RAM to run.

## How to run this project:

There are 2 possible ways - to use the audio_features.csv we added to the submission folder or to create it with our code:

With the use of our pre-made audio_features.csv:
1. start with the above enviroment setup instruction.
2. run the simple_classifyers.ipynb notebook.
3. run the RNN.ipynb notebook.
4. run the CNN.ipynb notebook.
5. run the transformer.ipynb.

Note: After running the code that creates the spectrogram (which is located in the RNN.ipynb) - step 3, the order doesn't matter.

Without the use of our pre-made audio_features.csv:
1. start with the above enviroment setup instruction.
2. run the data_loading.ipynb notebook.
3. run the audio_features.ipynb notebook.
4. run the simple_classifyers.ipynb notebook.
5. run the RNN.ipynb notebook.
6. run the CNN.ipynb notebook.
7. run the transformer.ipynb.

Note: After running the code that creates the spectrogram (which is located in the RNN.ipynb) - step 5, the order doesn't matter.
