from project.data_iterator import dataIterator
import torch

def dataIteratorTest():
    X, y = torch.rand((10,5)), torch.rand(10)
    loader = dataIterator(X,y,2)
    loader = iter(loader)
    print(list(loader))


if __name__ == "__main__":
    dataIteratorTest()