import torchaudio
import torch
from torchaudio import transforms
from pychorus.helpers import find_and_output_chorus

def rechannel(aud, new_channel = 1):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
        return aud

    if (new_channel == 1):
        resig = torch.mean(sig, dim=0).reshape(1,-1)
    else:
        resig = torch.cat([sig, sig])

    return ((resig, sr))


def get_chorus(path, duration, aud):
	sig, sr = aud
	start = find_and_output_chorus(path, None, duration, True)
	if start is None:
		start = sig.shape[-1]/sr * 0.4
	start = int(start)
	sig = sig[:,start*sr:(start+duration)*sr]
	return ((sig, sr))


def createSpect(aud, n_mels=128, n_fft=400, win_length = 400, hop_len=None):
    sig,sr = aud
    top_db = 80
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, win_length=win_length, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)
