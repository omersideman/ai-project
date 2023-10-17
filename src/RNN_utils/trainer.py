import torch
from tqdm import tqdm

class trainer():
    def __init__(self, model, loss_func, optimizer, scheduler, device):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device=device
        self.results = {'loss': [], 'accuracy': []}

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        y_prob = self.model(X)
        rows = y.shape[0]
        target = torch.zeros((rows,2),device=self.device)
        target[torch.arange(rows), y] = 1
        loss = self.loss_func(y_prob,target)
        loss.backward()
        self.optimizer.step()
        return loss.item(), torch.sum(torch.argmax(y_prob,dim=1)==y).item() 

    def train_epoch(self, train_dl, epoch, verbose = False):
        tot_corr = 0.0
        tot_loss = 0.0
        num_train = 0
        if verbose:
            for (X,y) in tqdm(iter(train_dl), desc='Train Batch'):
                loss, corr = self.train_batch(X,y)
                tot_loss += loss
                tot_corr += corr
                num_train += y.shape[0]
        else:
            for (X,y) in iter(train_dl):
                loss, corr = self.train_batch(X,y)
                tot_loss += loss
                tot_corr += corr
                num_train += y.shape[0]
        
        self.results['loss'].append(tot_loss)
        self.results['accuracy'].append(tot_corr/num_train)
        if verbose:
            print(f'Epoch #{epoch}: Loss - {tot_loss}, Accuracy - {tot_corr/num_train}')

    def train(self, train_dl, epochs=30, verbose = True):
        if not verbose:
            for epoch in tqdm(range(epochs)):
                self.train_epoch(train_dl,epoch, verbose)
                self.scheduler.step()
            loss, acc = self.results['loss'][-1], self.results['accuracy'][-1]
            print(f'Epoch #{epoch}: Loss - {loss}, Accuracy - {acc}')
        else:
            for epoch in range(epochs):
                self.train_epoch(train_dl,epoch, verbose)
                self.scheduler.step()
        return self.results

    def evaluate(self, dl, verbose = False):
        tot_corr = 0.0
        tot_loss = 0.0
        num_samples = 0
        if verbose:
            for (X,y) in tqdm(iter(dl), desc='Test Batch'):
                rows = y.shape[0]
                target = torch.zeros((rows,2),device=self.device)
                target[torch.arange(rows), y] = 1
                with torch.no_grad():
                    y_prob = self.model(X)
                tot_loss += self.loss_func(y_prob,target).item()
                tot_corr += torch.sum(torch.argmax(y_prob,dim=1)==y).item()
                num_samples += y.shape[0]
        else:
            for (X,y) in iter(dl):
                rows = y.shape[0]
                target = torch.zeros((rows,2),device=self.device)
                target[torch.arange(rows), y] = 1
                with torch.no_grad():
                    y_prob = self.model(X)
                tot_loss += self.loss_func(y_prob,target).item()
                tot_corr += torch.sum(torch.argmax(y_prob,dim=1)==y).item()
                num_samples += y.shape[0]
        if verbose:
            print(f'Val results: Loss - {tot_loss}, Accuracy - {tot_corr/num_samples}')

        return tot_loss, tot_corr/num_samples
            