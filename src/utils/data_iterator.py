import torch


class dataIterator:
    def __init__(self, X, y, batch, shuffle = False):
        self.X = X
        self.y = y
        self.bsize = batch
        self.start = 0
        self.end = batch - 1
        self.shuffle = shuffle

    def __iter__(self):
        self.perm = torch.randperm(self.X.shape[0]) if self.shuffle else torch.arange(end = self.X.shape[0])
        return self

    def __next__(self):
        if self.start < self.X.shape[0]:
            end = self.end+1 if self.end < self.X.shape[0] else self.X.shape[0]
            curr_perm = self.perm[self.start:end]
            self.start, self.end = self.end + 1, self.end + self.bsize
            return self.X[curr_perm], self.y[curr_perm]
        else:
            raise StopIteration