import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class PopulationDataset(Dataset):
    def __init__(self, room_csv_dir, product_family_label_csv, max_models=50):
        self.room_df = pd.DataFrame()
        self.max_models = max_models

        self.all_labels = [
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

    def get_label_from_index(self, index):
        return self.all_labels[index]

    def get_all_labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.csvFiles)

    def __getitem__(self, idx):
        roomCsv = pd.read_csv(self.room_csv_dir + "/" + self.csvFiles[idx])
        ## keep only the first 200 models + 1
        if (roomCsv.shape[0] > 50):
            print("rm " + self.room_csv_dir + "/" + self.csvFiles[idx] + " && ", end="")
        roomCsv = roomCsv.iloc[:self.max_models + 1, :]

        np_bb_sample = np.zeros((self.max_models * 8 + 8))
        np_bb_sample[:roomCsv.shape[0] * 8] = roomCsv.iloc[:, 1:9].to_numpy().flatten()

        np_bb_target = np.zeros((self.max_models * 8 + 8))
        np_bb_target[:roomCsv.shape[0] * 8] = roomCsv.iloc[:, 9:17].to_numpy().flatten()

        input_labels = np.zeros((self.max_models, len(self.all_labels)))
        # print(input_labels.shape)

        for index, label in enumerate(pd.Categorical(roomCsv.in_product_family.values)[1:]):
            input_labels[index, self.all_labels.index(label)] = 1.  # TODO FIX label index

        label_sample = input_labels.flatten()

        # sample is concatenation of npm_sample and label_sample

        minx_ = np.min(np_bb_sample[:8])
        max_ = np.max(np_bb_sample[:8])

        np_bb_sample_norm = (np_bb_sample - minx_) / (max_ - minx_)
        np_bb_target_norm = (np_bb_target - minx_) / (max_ - minx_)

        position_input = torch.tensor(np_bb_sample_norm, dtype=torch.float32)
        object_types = torch.tensor(label_sample, dtype=torch.float32)

        np_bb_sample_norm_x = np_bb_target_norm[::2]
        np_bb_sample_norm_y = np_bb_target_norm[1::2]

        reversed_value_np_bb_sample_norm_x = np.max(np_bb_sample_norm_x) - np_bb_sample_norm_x + np.min(np_bb_sample_norm_x)
        reversed_value_np_bb_sample_norm_y = np.max(np_bb_sample_norm_y) - np_bb_sample_norm_y + np.min(np_bb_sample_norm_y)

        # print(reversed_value_np_bb_sample_norm_x)
        # print(reversed_value_np_bb_sample_norm_y)

        # # intercalate (np_bb_sample_norm_x, reversed_value_np_bb_sample_norm_y) to get the reversed sample

        # axis symetry

        out1 = np.zeros((self.max_models * 8 + 8))
        out1[::2] = np_bb_sample_norm_x
        out1[1::2] = np_bb_sample_norm_y

        out2 = np.zeros((self.max_models * 8 + 8))
        out2[::2] = reversed_value_np_bb_sample_norm_x
        out2[1::2] = reversed_value_np_bb_sample_norm_y

        out3 = np.zeros((self.max_models * 8 + 8))
        out3[::2] = np_bb_sample_norm_x
        out3[1::2] = reversed_value_np_bb_sample_norm_y

        out4 = np.zeros((self.max_models * 8 + 8))
        out4[::2] = reversed_value_np_bb_sample_norm_x
        out4[1::2] = np_bb_sample_norm_y

        # ## 90° rotation
        # s = np.sin(np.pi / 2)
        # c = np.cos(np.pi / 2)

        # rotationed2 = np.zeros((self.max_models * 8 + 8))
        # rotationed2[::2] = np_bb_sample_norm_x * c - np_bb_sample_norm_y * s
        # rotationed2[1::2] = np_bb_sample_norm_x * s + np_bb_sample_norm_y * c

        # ## 180 rotation
        # s = np.sin(np.pi)
        # c = np.cos(np.pi)

        # rotationed3 = np.zeros((self.max_models * 8 + 8))
        # rotationed3[::2] = np_bb_sample_norm_x * c - np_bb_sample_norm_y * s
        # rotationed3[1::2] = np_bb_sample_norm_x * s + np_bb_sample_norm_y * c

        # # rotation
        # out5 = np.zeros((self.max_models * 8 + 8))
        # out5[::2] =
        # print(reversed_value_np_bb_sample_norm_x)
        # print(reversed_value_np_bb_sample_norm_y)
        ## rebuild a bb_target_norm with np_bb_sample_norm_x and reversed_value_np_bb_sample_norm_y

        position_output1 = torch.tensor(out1, dtype=torch.float32)
        position_output2 = torch.tensor(out2, dtype=torch.float32)
        position_output3 = torch.tensor(out3, dtype=torch.float32)
        position_output4 = torch.tensor(out4, dtype=torch.float32)

        position_output = torch.cat((position_output1, position_output2, position_output3, position_output4), 0)
        sample = torch.cat((position_input, object_types), 0)

        return sample, position_output