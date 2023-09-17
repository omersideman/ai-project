import torch
from copy import deepcopy
from src.RNN_utils.trainer import trainer
from sklearn.model_selection import KFold

SEED = 42

class crossValidate:
    def __init__(self, train_ds, folds = 5, batch_size = 32):
        self.batch_size = batch_size
        self.folds = folds
        self.train_dls = []
        self.val_dls = []
        kf = KFold(n_splits=folds, random_state=SEED, shuffle=True)
        for (train_idx, test_idx) in kf.split(train_ds):
            self.train_dls.append(
                            torch.utils.data.DataLoader(
                                    dataset=train_ds,
                                    batch_size=batch_size,
                                    sampler=torch.utils.data.SubsetRandomSampler(train_idx))
                            )
            self.val_dls.append(
                            torch.utils.data.DataLoader(
                                    dataset=train_ds,
                                    batch_size=batch_size,
                                    sampler=torch.utils.data.SubsetRandomSampler(test_idx))
                            )

    def runCV(self, trainer: trainer, epochs = 40):
        train_results = {'loss': [0.0 for e in range(epochs)], 'accuracy': [0.0 for e in range(epochs)]}
        val_results = {'loss': [0.0 for e in range(epochs)], 'accuracy': [0.0 for e in range(epochs)]}
        add_func = lambda a,b: a + b
        div_func = lambda x: x / self.folds
        for fold, (train_dl,val_dl) in enumerate(zip(self.train_dls,self.val_dls)):
            print(f'Fold #{fold}:')
            trainer_cp = deepcopy(trainer)
            val_loss, val_acc = [], []
            for epoch in range(epochs):
                trainer_cp.train_epoch(train_dl,epoch,True)
                loss, acc = trainer_cp.evaluate(val_dl, True)
                val_loss.append(loss); val_acc.append(acc)
            
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