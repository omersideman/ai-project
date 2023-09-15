from torch.utils.data import Dataset
import torch

"""
class SoundDS(Dataset):
	def __init__(self, df, data_path, channel = 1, ext = '.mp3', duration = 30, n_mel=128, n_fft=400,win_length=400):
		self.df = df
		self.data_path = str(data_path)
		self.channel = channel
		self.ext = ext
		self.duration = duration
		self.n_mel = n_mel
		self.n_fft = n_fft
		self.win_length = win_length
	
	def __len__(self):
		return len(self.df)    

	def __getitem__(self, idx):
		song_path = self.data_path + '/' + self.df.loc[idx,'id'] + self.ext
		class_id = self.df.loc[idx, 'viral']

		#load the audio file
		aud = torchaudio.load(song_path)

		#convert the audio to mono audio
		aud = rechannel(aud,new_channel=1)

		#take only the part of the chorus from the signal
		aud = get_chorus(song_path, self.duration, aud)

		#create the mel-spectogram
		sgram = createSpect(aud, n_mels=self.n_mel, n_fft=self.n_fft, win_length=self.win_length)
		return sgram, class_id
"""

class SoundDS(Dataset):
	def __init__(self, df, data_path):
		self.df = df
		self.indices =  df.index
		self.data_path = str(data_path)
	
	def __len__(self):
		return len(self.df)    

	def __getitem__(self, idx):
		song_path = self.data_path + '/' + self.df.loc[idx,'id'] + '.pt'
		class_id = self.df.loc[self.indices[idx], 'viral']

		#load the spectorgram tensor file
		sgram = torch.load(song_path)
		return sgram, class_id