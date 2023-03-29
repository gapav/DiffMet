import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
from os import listdir
from PIL import Image

import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class RadarPrecipitationSequence(Dataset):
    def __init__(
        self,
        root_dir,
        num_cond_frames,
        frames_to_predict,
        img_out_size,
        prediction_time_step_ahead,
        train_test_val,
        train_fname=None,
        test_fname=None,
        transform=None,
        emb_transform=None,
        print_fname=False,
        center_crop=False,
        nowcast_mode=False,
    ):
        """
        Args:

        """
        self.root_dir = root_dir
        self.train_test_val = train_test_val

        if self.train_test_val == "train":
            self.root_dir = f"{self.root_dir}/train"

        elif self.train_test_val == "test":
            self.root_dir = f"{self.root_dir}/test"

        elif self.train_test_val == "val":
            self.root_dir = f"{self.root_dir}/val"
        else:
            print("provide train,test or val as argument when init dataset")
            return

        self.center_crop = center_crop
        self.num_cond_frames = num_cond_frames
        self.frames_to_predict = frames_to_predict
        self.img_out_size = img_out_size
        self.prediction_time_step_ahead = prediction_time_step_ahead
        self.transform = transform
        self.emb_transform = emb_transform
        self.print_fname = print_fname
        self.filenames = listdir(self.root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Function to load data sequence from folder to model. Creates a random crop of size
        self.img_out_size x self.img_out_size. Returns 3 arrays, one for lwe, one for conv/stratiform and one for time. 

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        fname = self.filenames[idx]
        array_path = os.path.join(self.root_dir, self.filenames[idx])
        if self.print_fname:
            print(array_path)
        sequence_array = np.load(array_path)
        if self.train_test_val == "train":
            lwe_sequence_array = sequence_array[:, :, :]

        elif self.train_test_val == "test" or self.train_test_val == "val":
            lwe_sequence_array = sequence_array[0, :, :, :]
            conv_strat_array = sequence_array[1, :, :, :]
        else:
            print("PROVIDE TRAIN/TEST/VAL")

        # conv_sequence_array = sequence_array[1, :, :, :]
        # time_array = sequence_array[2, :, :, :]

        lwe_sequence_array = np.squeeze(lwe_sequence_array)

        cond_emb_transformed = torch.empty(
            (self.num_cond_frames, 224, 224)  # 224 = minimum input size for resnet18
        )

        input_shape = lwe_sequence_array.shape[2]

        if self.center_crop:
            idx = 96
            idy = 96
        else:
            idx = np.random.randint(0, (input_shape - (self.img_out_size + 1)))
            idy = np.random.randint(0, (input_shape - (self.img_out_size + 1)))

        cond_emb_imgs = lwe_sequence_array[
            0 : self.num_cond_frames,
            idx : idx + self.img_out_size,
            idy : idy + self.img_out_size,
        ]

        for i in range(len(cond_emb_imgs)):

            img = self.emb_transform(cond_emb_imgs[i])
            cond_emb_transformed[i, :, :] = img

        img = lwe_sequence_array[
            (self.num_cond_frames + self.prediction_time_step_ahead) - 1,
            idx : idx + self.img_out_size,
            idy : idy + self.img_out_size,
        ]

        img = self.transform(img)

        # crop conv_seq_array to same size and indicies as lwe
        # conv_sequence_array = conv_sequence_array[
        #     :, idx : idx + self.img_out_size, idy : idy + self.img_out_size,
        # ]
        # time array only contains the time info for each frame in idx 0,0:
        # time_array = time_array[:, 0 : self.img_out_size, 0 : self.img_out_size]
        if self.train_test_val == "train":
            return img, cond_emb_transformed, fname, idx, idy
        else:
            lwe_tensor = torch.empty((8, 64, 64))
            lwe_sequence_array_sliced = lwe_sequence_array[
                :, idx : idx + self.img_out_size, idy : idy + self.img_out_size,
            ]

            for i in range(lwe_sequence_array_sliced.shape[0]):
                lwe_img = self.transform(lwe_sequence_array_sliced[i])
                lwe_tensor[i, :, :] = lwe_img

            conv_strat_crop = conv_strat_array[
                :, idx : idx + self.img_out_size, idy : idy + self.img_out_size,
            ]

            return (
                img,
                cond_emb_transformed,
                conv_strat_crop,
                lwe_tensor,
                fname,
                idx,
                idy,
            )
