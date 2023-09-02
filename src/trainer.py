import torch
from tqdm import tqdm
from src.utils.data_iterator import  dataIterator

class trainer:
    def __init__(self, model, optimizer, loss_fn, epochs, batch, scheduler = None, shuffle = False, verbose = False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.bsize = batch
        self.scheduler = scheduler
        self.shuffle = shuffle
        self.verbose = verbose

    def fit(self, X_train, y_train, X_val = None, y_val = None):
        res_train = [[],[]]
        res_val = [[],[]]
        if not self.scheduler is None:
            sched_state = self.scheduler.state_dict()
        for epoch in tqdm(range(self.epochs)):
            loader = dataIterator(X_train, y_train, self.bsize, self.shuffle)
            loss, corr = self.train_epoch(loader)
            res_train[0].append(loss)
            res_train[1].append(corr/X_train.shape[0])
            if self.verbose or epoch== self.epochs - 1:
                print(f'Epoch #{epoch}: Loss={res_train[0][-1]}, accuracy={res_train[1][-1]}')
            
            if not self.scheduler is None:
                self.scheduler.step()

            if X_val is None:
                continue

            #For cross validation
            loss, acc = self.evaluate(X_val, y_val)
            res_val[0].append(loss)
            res_val[1].append(acc)

        if not self.scheduler is None:
            self.scheduler.load_state_dict(sched_state)
        
        if X_val is None:
            return res_train
        else:
            return res_train, res_val

    def train_epoch(self, loader):
        total_loss = 0.
        total_correct = 0.
        for X_batch, y_batch in iter(loader):
            loss, corr = self.train_batch(X_batch, y_batch)
            total_loss += loss
            total_correct += corr
        return total_loss, total_correct

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        corr = (torch.argmax(y_pred,dim=1) == y).type(torch.LongTensor).sum()
        return loss.item(), corr.item()

    def evaluate(self, X, y):
        with torch.no_grad():
            y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        acc = (torch.argmax(y_pred,dim=1) == y).type(torch.FloatTensor).mean()
        return loss.item(), acc.item()