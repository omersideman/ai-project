import os
import torch
from torch.utils.data import Dataset

class SoundDS(Dataset):
  def __init__(self, df, data_path, preprocess):
    self.df = df
    self.indices =  df.index
    self.data_path = str(data_path)
    self.preprocess = preprocess

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    # song_path = self.data_path + '/' + self.df.loc[idx,'id'] + '.pt'
    song_path = os.path.join(self.data_path, self.df.loc[idx,'id'] + '.pt')
    if not os.path.exists(song_path):
      print("File not found: ", song_path)
      raise FileNotFoundError(f"File not found: {song_path}")
    
    class_id = self.df.loc[self.indices[idx], 'viral']

    #load the spectorgram tensor file
    sgram = torch.load(song_path)

    sgram = self.preprocess(sgram)

    return sgram, class_id

class SimpleDS(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return (self.X).shape[0]

	def __getitem__(self, idx):
		return self.X[idx,:], self.y[idx]