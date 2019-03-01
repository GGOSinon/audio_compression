import librosa
import numpy as np
import math
import cmath

def amplitude_to_db(x):
	if x<0.001: return -30
	return 10 * math.log(x)

def normalize(x):
	return (x-x.min())/(x.max()-x.min())

def getSTFT(wavfile, n_fft, n_hop):
	stft_data = librosa.stft(wavfile, n_fft = n_fft, hop_length = n_hop)
	k, m = stft_data.shape[0], stft_data.shape[1]
	A = np.zeros((k, m))
	for i in range(k):
		for j in range(m):
			A[i][j] = amplitude_to_db(abs(stft_data[i][j]))
	A = np.transpose(A)
	return A

import os
import random

file_list = []
for root, dirs, files in os.walk("../../source_seperation/DSD100/Mixtures"):
	for file in files:
		if file.endswith("mixture.wav"):
			file_list.append(os.path.join(root, file))

file_num = 1
n_fft, n_hop = 1024, 512
data_size = n_fft//2 + 1
# n_fft x n_fft size image

num_channel = 2
num_dataset = 500
dataset = []

for i in range(file_num):
	y, sr = librosa.load(file_list[i])
	stft = getSTFT(y, n_fft = n_fft, n_hop = n_hop)
	print(stft.shape)
	len_music = stft.shape[0]
	for k in range(num_dataset):
		if k % 10 == 0: print("File %d: %d" % (i, k))
		pos = random.randrange(0, len_music - data_size)
		data = stft[pos : pos + data_size]
		data = normalize(data)
		#print(data.min(), data.max())
		dataset.append(data)
		

dataset = np.array(dataset)
print(dataset.shape)
np.save("data.npy", dataset)
