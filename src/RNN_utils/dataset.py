from torch.utils.data import Dataset
import torch

class SoundDS(Dataset):
  def __init__(self, df, data_path, channels=1, cls=False):
    self.df = df
    self.indices =  df.index
    self.data_path = str(data_path)
    self.channels = channels
    self.cls = cls

  def __len__(self):
    return len(self.df)    

  def __getitem__(self, idx):
    song_path = self.data_path + '/' + self.df.loc[idx,'id'] + '.pt'
    class_id = self.df.loc[self.indices[idx], 'viral']  

    #load the spectorgram tensor file
    sgram = torch.load(song_path)
    _, rows, columns = sgram.shape  

    # Only if there is only one channel, remove the dimension of the channels
    if self.channels == 1:
      sgram = sgram.reshape(rows, columns)
    
    # Change between the time dimension and the features dimension as LSTM/TransformerEncoder requires
    sgram = torch.transpose(sgram,dim0=-2,dim1=-1)
    
    if self.cls:
      sgram = torch.cat((torch.zeros(sgram.shape[:-1]).unsqueeze(-1), sgram), dim=-1)
    return sgram, class_id
	
class SimpleDS(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return (self.X).shape[0]
	
	def __getitem__(self, idx):
		return self.X[idx,:], self.y[idx]