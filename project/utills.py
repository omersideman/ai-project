import matplotlib.pyplot as plt
import numpy as np

def plotCV(results, configures, size = (15,10)):
    fig, axs = plt.subplots(2,2, figsize=size)
    num_epochs = len(results[0][0][0])

    for i,config in enumerate(configures):
        for j in range(2):
            axs[j][0].plot(range(num_epochs), results[i][0][j], label=f'{config}')
            axs[j][0].set_xlabel('epoch')
            axs[j][0].set_ylabel('loss')

            axs[j][1].plot(range(num_epochs), results[i][1][j], label=f'{config}')
            axs[j][1].set_xlabel('epoch')
            axs[j][1].set_ylabel('accuracy')

    plt.legend()