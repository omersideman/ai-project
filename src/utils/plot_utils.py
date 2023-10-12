import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import numpy as np

colors = ["#016A70", "#D2DE32", '#FF6969']


def set_style(ax):
    sns.despine(ax=ax, left=True)
    ax.grid(axis='y', linewidth=0.3, color='black')


def hist(df, x, ax, main_color=colors[1], second_color=colors[0], bins=30):

    sns.histplot(data=df, x=x, bins=bins, ax=ax,  # type: ignore
                 kde=True, color=main_color,
                 edgecolor=second_color, line_kws={"linestyle": '--'}, linewidth=1
                 )
    ax.lines[0].set_color(second_color)
    set_style(ax)

    ax.set_xlabel(x.replace("_", " ").capitalize(), fontsize="large")
    ax.set_ylabel("")


def count(df, x, ax, main_color=colors[1], second_color=colors[0]):

    ax.bar(df[x].value_counts().index, df[x].value_counts().values,
           color=main_color, edgecolor=second_color, linewidth=3)

    set_style(ax)

    ax.set_xlabel(x.replace("_", " ").capitalize(), fontsize="x-large")
    ax.set_ylabel("")


def scatter(df, x, y, ax, main_color=colors[1], second_color=colors[0]):

    sns.regplot(data=df, x=x, y=y, ax=ax,
                color=main_color, ci=75,
                scatter_kws={
                    'edgecolor': second_color,
                    'linewidth': 1.5,
                    's': 50
                },
                line_kws={
                    'color': colors[2],
                    'linewidth': 3,
                }
                )
    ax.set_xlabel(x.replace("_", " ").capitalize())
    ax.set_ylabel(y.replace("_", " ").capitalize())

    sns.despine(ax=ax)
    ax.grid(axis='x')


def stripplot(df, x, y, ax, palette=[colors[1], colors[2]]):

    sns.stripplot(data=df, x=x, y=y, palette=palette, ax=ax,
                  linewidth=2, size=8)

    set_style(ax)


def plotCV(results, configures, size=(15, 10)):
    fig, axs = plt.subplots(2, 2, figsize=size)
    num_epochs = len(results[0][0][0])

    for (config, result) in zip(configures, results):
        for j in range(2):
            axs[j][0].plot(range(num_epochs), result[j][0], label=f'{config}')
            axs[j][0].set_xlabel('epoch')
            axs[j][0].set_ylabel('loss')

            axs[j][1].plot(range(num_epochs), result[j][1], label=f'{config}')
            axs[j][1].set_xlabel('epoch')
            axs[j][1].set_ylabel('accuracy')

    plt.legend()


def plot_audio_wave(wave):
    plt.plot(wave)
    plt.title("Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")


def plot_mel_spectrogram(wave, sr, mel=True):
    # Computing the mel spectrogram
    spect = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=2048, hop_length=512)
    spect_db = librosa.power_to_db(spect, ref=np.max)  # converting to decibals
    #
    # Plotting the spectrogram
    plt.figure(figsize=(8, 5))
    librosa.display.specshow(spect_db, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")

    plt.title("Mel Spectrogram")
    print(f"Mel Spectrogram shape: {spect.shape}")
