import pandas as pd
import torch
import torchmetrics as torchmetrics
from matplotlib import pyplot, pyplot as plt
from sklearn.model_selection import *
from torch._C._monitor import data_value_t
from torch.optim import Optimizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
import math
import random as rn
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import torch.cuda
from sklearn.preprocessing import StandardScaler



#set Seed for repr.
seed = 1208
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rn.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
print(f"Using {device} device")

#Sequence Lenght
SEQUENCE_LEN = 4

#Batch Size
BATCH_SIZE = 128
#Shuffle
SHUFFLE = False
# CreateTrainDataset Read Data
class Normal_Dataset(Dataset):

    def __init__(self, set, drop_first=False):


        df = pd.read_pickle('./data/storedDF/neu/normalV1.pkl')
        if drop_first:
            df = df.drop(df.index[:21600])

        df_timestamp = df.copy()
        del df["Timestamp"]
        dfy = df["Class"]
        y_all = dfy.to_numpy(dtype=np.int64)
        del df["Class"]

        #x_all = df.to_numpy(dtype=np.float64) #macht keinen Unterschied
        x_all = df.to_numpy()
        pipeline = Pipeline([('normalizer', StandardScaler())])
        #pipeline = Pipeline([('normalizer', MinMaxScaler())])
        pipeline.fit(x_all)
        x_all = pipeline.transform(x_all)
        # Split in Test and Train
        X_train, X_val, Y_train, Y_val = train_test_split(x_all, y_all, test_size=0.2, shuffle=SHUFFLE)# , random_state=22)


        # Anwenden auf den Datensaetzen (Training und Validation)
        #X_train_preprocessed = pipeline.transform(X_train)
        X_train_preprocessed = torch.from_numpy(X_train)

        X_val_preprocessed = torch.from_numpy(X_val)

        self.y_val = torch.from_numpy(Y_val)
        self.x_val = X_val_preprocessed

        self.x_train = X_train_preprocessed

        self.y_train = torch.from_numpy(Y_train)

        self.pipeline = pipeline
        self.n_samples = x_all.shape[0]

        self.df_timestamp = df_timestamp
        self.x_all = x_all

        self.set = set


dataset_normal = Normal_Dataset(drop_first=True, set="x_train")
print("read normal done!")


# Create Attacked Dataset
class Attacked_Dataset(Dataset):

    def __init__(self, pipeline, set):
        df = pd.read_pickle('./data/storedDF/neu/attackV0.pkl')

        df_timeStamp = df.copy()
        del df["Timestamp"]

        df_both = df.copy()
        dfy_both = df_both["Class"]
        print(f'Normal 0 and Attacked 1 in Attacked: (not in window Shape) \n {dfy_both.value_counts()}')

        y_both = dfy_both.to_numpy(dtype=np.int32)
        del df_both["Class"]

        x_both = df_both.to_numpy()

        x_both = pipeline.transform(x_both)

        self.x_full_attacked_inc_normal = torch.from_numpy(x_both)
        self.y_full_attacked_inc_normal = torch.from_numpy(y_both)

        self.df_timestamp = df_timeStamp
        self.set = set
        #self.sequence_length = seq_len



dataset_attacked = Attacked_Dataset(dataset_normal.pipeline, set="x_full_attacked_inc_normal")




class WindowDataset(Dataset):
    def __init__(self, data, label, window):
        self.data = data
        self.window_size = window
        self.count = 0
        self.label = label
    def __getitem__(self, index):
        self.count += 1
        x = self.data[index:index+self.window_size]
        index_label_last = index+self.window_size-1
        index_label_first = index
        last_label = self.label[index+self.window_size-1]
        first_label = self.label[index]

        label = 0
        for l in self.label[index:index + self.window_size - 1]:
            if l == 1:
                label = 1
                break
        first_occurence = label

        return x, last_label

    def __len__(self):
        return (self.data.shape[0]-self.window_size +1)

train_dataset_window = WindowDataset(dataset_normal.x_train, dataset_normal.y_train, SEQUENCE_LEN)
validation_dataset_window = WindowDataset(dataset_normal.x_val, dataset_normal.y_val, SEQUENCE_LEN)
attacked_test_dataset_window = WindowDataset(dataset_attacked.x_full_attacked_inc_normal, dataset_attacked.y_full_attacked_inc_normal, SEQUENCE_LEN)

train_dataloader_window = DataLoader(train_dataset_window, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
validation_dataloader_window = DataLoader(validation_dataset_window, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
attacked_test_dataloader_window = DataLoader(attacked_test_dataset_window, batch_size=BATCH_SIZE,shuffle=SHUFFLE)


print("Read Done!")

