from librosa.core.audio import load
import numpy as np
import pandas as pd
import os
from librosa import stft
from pandas.errors import EmptyDataError
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def load_and_preprocess_data(dataset_root_folder="dataset"):
  folders = [folder for folder in os.listdir(dataset_root_folder)]
  stfts = []
  labels = []
  bad_files = []
  for folder in folders:
    print(f"Processing {folder}")
    for file in tqdm(os.listdir((f"{dataset_root_folder}/{folder}"))):
      channel_stfts = []
      try:
        signal = pd.read_csv(f"{dataset_root_folder}/{folder}/{file}",skiprows=6)
        signal = signal.iloc[251:1001,2:4].T
        if signal.shape[1] < 750:
          continue
        signal= signal.to_numpy()
        for channel in signal:
          channel_stft = stft(channel,n_fft=250,hop_length=125)
          channel_stfts.append(channel_stft)
        stfts.append(np.array(channel_stfts))
        labels.append(1 if folder == "single_blink" else 0 )
      except EmptyDataError:
        bad_files.append(file)
  print(f"Bad files {bad_files}")
  return stfts,labels


def split_dataset(features,labels,test_size=0.2):
  x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=test_size)
  return x_train,x_test,y_train,y_test

stfts,labels = load_and_preprocess_data()
print(stfts[0].shape)
x_train,x_test,y_train,y_test = split_dataset(stfts,labels)