import torch
from copy import deepcopy
from src.RNN_utils.trainer import trainer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import default_collate

SEED = 42

class crossValidate:
    def __init__(self, train_ds, device, folds = 5, batch_size = 32):
        self.batch_size = batch_size
        self.folds = folds
        self.train_dls = []
        self.val_dls = []
        self.device = device
        kf = KFold(n_splits=folds, random_state=SEED, shuffle=True)
        to_gpu = lambda x: list(map(lambda t: t.to(self.device), default_collate(x)))
        for (train_idx, test_idx) in kf.split(train_ds):
            self.train_dls.append(
                            torch.utils.data.DataLoader(
                                    dataset=train_ds,
                                    batch_size=batch_size,
                                    sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                                    collate_fn = to_gpu)
                            )
            self.val_dls.append(
                            torch.utils.data.DataLoader(
                                    dataset=train_ds,
                                    batch_size=batch_size,
                                    sampler=torch.utils.data.SubsetRandomSampler(test_idx),
                                    collate_fn = to_gpu)
                            )

    def runCV(self, trainer: trainer, epochs = 40, verbose = True):
        train_results = {'loss': [0.0 for e in range(epochs)], 'accuracy': [0.0 for e in range(epochs)]}
        val_results = {'loss': [0.0 for e in range(epochs)], 'accuracy': [0.0 for e in range(epochs)]}
        add_func = lambda a,b: a + b
        div_func = lambda x: x / self.folds
        for fold, (train_dl,val_dl) in enumerate(zip(self.train_dls,self.val_dls)):
            print(f'Fold #{fold}:')
            trainer_cp = deepcopy(trainer)
            val_loss, val_acc = [], []
            for epoch in range(epochs):
                trainer_cp.train_epoch(train_dl,epoch, verbose)
                loss, acc = trainer_cp.evaluate(val_dl, verbose)
                val_loss.append(loss); val_acc.append(acc)
            if not verbose:
                print(f'Result: ',{'loss_train':trainer_cp.results['loss'][-1], 'accuracy_train': trainer_cp.results['accuracy'][-1], 'loss_test': loss, 'accuracy_test': acc})
            train_loss, train_acc = trainer_cp.results['loss'], trainer_cp.results['accuracy']
            
            # adding the training fold data
            train_results['loss'] = list(map(add_func, train_results['loss'], train_loss))
            train_results['accuracy'] = list(map(add_func, train_results['accuracy'], train_acc))

            # adding the validation fold data
            val_results['loss'] = list(map(add_func, val_results['loss'], val_loss))
            val_results['accuracy'] = list(map(add_func, val_results['accuracy'], val_acc))
        
        train_results['loss'] = list(map(div_func, train_results['loss']))
        train_results['accuracy'] = list(map(div_func, train_results['accuracy']))
        val_results['loss'] = list(map(div_func, val_results['loss']))
        val_results['accuracy'] = list(map(div_func, val_results['accuracy']))

        return (train_results, val_results)
    
def plotCV(results, configures, size = (15, 10), title= 'CV results'):
    fig, axs = plt.subplots(2, 2, figsize=size)
    num_epochs = len(results[0][0]['loss'])

    for (config, result) in zip(configures, results):
        for j in range(2):
            axs[j][0].plot(range(num_epochs), result[j]['loss'], label=f'{config}')
            axs[j][0].set_xlabel('epoch')
            axs[j][0].set_ylabel('loss')

            axs[j][1].plot(range(num_epochs), result[j]['accuracy'], label=f'{config}')
            axs[j][1].set_xlabel('epoch')
            axs[j][1].set_ylabel('accuracy')

    fig.suptitle(title)
    axs[0][0].title.set_text('train loss')
    axs[0][1].title.set_text('train accuracy')
    axs[1][0].title.set_text('test loss')
    axs[1][1].title.set_text('test accuracy')
    plt.legend()
    plt.show()