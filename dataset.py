import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class PopulationDataset(Dataset):
    def __init__(self, room_csv_dir, product_family_label_csv, max_models=50):
        self.room_df = pd.DataFrame()
        self.max_models = max_models

        self.all_labels =[
            'RADIATOR',
            'ARMCHAIR',
            'VEGETATION',
            'TABLE',
            'LIGHTING',
            'STORAGE_FURNITURE',
            'DECORATION',
            'SOFA',
            'SEATING',
        ]

        # print(len(self.product_family_label_csv))
        self.room_csv_dir = room_csv_dir
        self.csvFiles = os.listdir(room_csv_dir)

    def __len__(self):
        return len(self.csvFiles)

    def __getitem__(self, idx):
        roomCsv = pd.read_csv(self.room_csv_dir + "/" + self.csvFiles[idx])
        ## keep only the first 200 models + 1
        if (roomCsv.shape[0] > 50):
            print(self.room_csv_dir + '/' + self.csvFiles[idx])
        roomCsv = roomCsv.iloc[:self.max_models + 1, :]

        np_sample = np.zeros((self.max_models * 8 + 8))
        np_sample[:roomCsv.shape[0] * 8] = roomCsv.iloc[:, 1:9].to_numpy().flatten()

        np_target = np.zeros((self.max_models * 8 + 8))
        np_target[:roomCsv.shape[0] * 8] = roomCsv.iloc[:, 9:17].to_numpy().flatten()

        input_labels = np.zeros((self.max_models, len(self.all_labels)))
        # print(input_labels.shape)

        for index, label in enumerate(pd.Categorical(roomCsv.in_product_family.values)[1:]):
            input_labels[index, self.all_labels.index(label)] = 1. # TODO FIX label index
        
        label_sample = input_labels.flatten()
        
        # sample is concatenation of npm_sample and label_sample

        sample = torch.nn.functional.normalize(torch.tensor(np_sample), dim=0)
        sample = torch.cat((sample, torch.tensor(label_sample)), 0)
        target = torch.nn.functional.normalize(torch.tensor(np_target), dim=0)

        return sample, target