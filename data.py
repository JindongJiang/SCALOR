import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from common import *
import re
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainStation(Dataset):
    def __init__(self, args, train=False):

        self.args = args
        self.img_h = img_h
        self.img_w = img_w
        self.object_act_size = object_act_size
        self.root = os.path.expanduser(self.args.data_dir)
        self.phase_train = train
        self.frame_skip = 8

        self.all_image_name_list = [os.path.join(self.root, s) for s in os.listdir(self.root) if s.endswith('.png')]
        self.all_image_name_list.sort(key=lambda s: int(re.split('/|\.', s)[-2]))

        if self.phase_train:
            self.all_image_name_list = self.all_image_name_list[:-(len(self.all_image_name_list) // 10)]
        else:
            self.all_image_name_list = self.all_image_name_list[-(len(self.all_image_name_list) // 10):]

        if self.phase_train:
            self.num_data = len(self.all_image_name_list) - 79
        else:
            # num of segments of data, segment is w.r.t. seq_len and
            self.num_data = (len(self.all_image_name_list) - 79) // seq_len

    def __getitem__(self, index):
        if self.phase_train:
            l = index // self.num_data
            index = index % self.num_data
        else:
            l = index // self.num_data
            index_for_each_scene = index % self.num_data
            index_segment = index_for_each_scene // self.frame_skip
            index_bias = index_for_each_scene % self.frame_skip
            index = index_segment * seq_len * self.frame_skip + index_bias

        # if not self.phase_train:
        #     index *= seq_len

        k = random.choice([self.frame_skip])
        # l = random.choice([0, 1, 2, 3, 4])
        image_list = []
        for i in range(seq_len):
            f_n = self.all_image_name_list[index + i * k]
            im = Image.open(f_n)
            width, height = im.size
            r = height / 2
            if l == 0:
                left_edge = r // 2
                upper_edge = 0
            elif l == 1:
                left_edge = r // 2 + r
                upper_edge = 0
            elif l == 2:
                left_edge = 0
                upper_edge = r
            elif l == 3:
                left_edge = r
                upper_edge = r
            elif l == 4:
                left_edge = r * 2
                upper_edge = r
            elif l == 5:
                left_edge = 0
                upper_edge = r // 2
            elif l == 6:
                left_edge = r
                upper_edge = r // 2
            elif l == 7:
                left_edge = r * 2
                upper_edge = r // 2

            im = im.crop(box=(left_edge, upper_edge, left_edge + self.args.train_station_cropping_origin,
                              upper_edge + self.args.train_station_cropping_origin))
            im = im.resize((img_h, img_w), resample=Image.BILINEAR)
            im_tensor = torch.from_numpy(np.array(im) / 255).permute(2, 0, 1)
            image_list.append(im_tensor)

        img = torch.stack(image_list, dim=0)

        return img.float(), torch.zeros(1)

    def __len__(self):
        return self.num_data * 8
