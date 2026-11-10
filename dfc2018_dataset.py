import os
import os.path
from osgeo import gdal
from torch.utils.data import Dataset
import numpy as np
import random
import cv2

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.arg = arg
        h,w = 512,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.bgr2rgb = bgr2rgb

        self.multi_dir = os.path.join(data_root, "train/multi")
        self.hyper_dir = os.path.join(data_root, "train/hyper")

        multi_list = [f for f in os.listdir(self.multi_dir)
                      if f.endswith(".tif") and not f.startswith("._")]
        multi_list.sort()

        self.multi_path = []
        self.hyper_path = []

        missing = 0
        for f in multi_list:
            multi_path = os.path.join(self.multi_dir, f)
            hyper_path = os.path.join(self.hyper_dir, f)
            if not os.path.exists(hyper_path):
                missing += 1
                continue
            self.multi_path.append(multi_path)
            self.hyper_path.append(hyper_path)

        print(f'len(multi) of DFC2018 Train dataset:{len(self.multi_path)}')

        self.img_num = len(self.hyper_path)
        self.length = self.patch_per_img * self.img_num  # 整个训练集的patch总数

    def arguement(self, img, rotTimes, vFlip, hFlip):
        for j in range(rotTimes):
            img_ = np.rot90(img.copy(), axes=(1, 2))
            img = img_.copy()
        for j in range(hFlip):
            img = img[:, :, ::-1].copy()
        for j in range(vFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size

        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line


        bgr = cv2.imread(self.multi_path[img_idx], cv2.IMREAD_UNCHANGED)
        if self.bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = bgr/255
        bgr = np.transpose(bgr, [2,0,1]) # [3,512,512]

        ds = gdal.Open(self.hyper_path[img_idx])
        hyper = ds.ReadAsArray()
        hyper = hyper.astype(np.float32)
        hyper = hyper/4095

        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper) # (image,label)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.bgr2rgb = bgr2rgb
        self.multi_dir = os.path.join(data_root, "test/multi")
        self.hyper_dir = os.path.join(data_root, "test/hyper")

        multi_list = [f for f in os.listdir(self.multi_dir)
                      if f.endswith(".tif") and not f.startswith("._")]
        multi_list.sort()

        self.multi_path = []
        self.hyper_path = []

        missing = 0
        for f in multi_list:
            multi_path = os.path.join(self.multi_dir, f)
            hyper_path = os.path.join(self.hyper_dir, f)
            if not os.path.exists(hyper_path):
                missing += 1
                continue
            self.multi_path.append(multi_path)
            self.hyper_path.append(hyper_path)

        print(f'len(hyper_test) of DFC2018 Valid dataset:{len(self.hyper_path)}')
        print(f'len(bgr_test) of DFC2018 Valid dataset:{len(self.multi_path)}')

    def __getitem__(self, idx):
        # 读multi
        bgr = cv2.imread(self.multi_path[idx], cv2.IMREAD_UNCHANGED)
        if self.bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = bgr/255
        bgr = np.transpose(bgr, [2, 0, 1])

        # 读hyper
        ds = gdal.Open(self.hyper_path[idx])
        hyper = ds.ReadAsArray()
        hyper = hyper.astype(np.float32)
        hyper =hyper/4095

        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hyper_path)

class TestDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.bgr2rgb = bgr2rgb
        self.multi_dir = os.path.join(data_root, "test/multi")
        self.hyper_dir = os.path.join(data_root, "test/hyper")

        multi_list = [f for f in os.listdir(self.multi_dir)
                      if f.endswith(".tif") and not f.startswith("._")]
        multi_list.sort()

        self.multi_path = []
        self.hyper_path = []

        missing = 0
        for f in multi_list:
            multi_path = os.path.join(self.multi_dir, f)
            hyper_path = os.path.join(self.hyper_dir, f)
            if not os.path.exists(hyper_path):
                missing += 1
                continue
            self.multi_path.append(multi_path)
            self.hyper_path.append(hyper_path)

        print(f'len(hyper_test) of DFC2018 Test dataset:{len(self.hyper_path)}')
        print(f'len(bgr_test) of DFC2018 Test dataset:{len(self.multi_path)}')


    def __getitem__(self, idx):
        bgr = cv2.imread(self.multi_path[idx], cv2.IMREAD_UNCHANGED)
        if self.bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = bgr/255
        bgr = np.transpose(bgr, [2, 0, 1])

        ds = gdal.Open(self.hyper_path[idx])
        hyper = ds.ReadAsArray()
        hyper = hyper.astype(np.float32)
        hyper = hyper/4095

        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hyper_path)