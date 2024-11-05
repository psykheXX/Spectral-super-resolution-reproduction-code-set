"""
reference repo https://github.com/caiyuanhao1998/MST-plus-plus
@inproceedings{mst_pp,
  title={MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction},
  author={Yuanhao Cai and Jing Lin and Zudi Lin and Haoqian Wang and Yulun Zhang and Hanspeter Pfister and Radu Timofte and Luc Van Gool},
  booktitle={CVPRW},
  year={2022}
}
"""
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import scipy.io as sio

class TrainDataset(Dataset):
    def __init__(self, patch_size, arg=True, bgr2rgb=True, stride=8):
        # 裁剪尺寸
        self.patch_size = patch_size
        # 存储高光谱图像数据的列表
        self.hypers = []
        # 存储rgb图像数据的列表
        self.RGBs = []
        # 是否进行数据增强
        self.arg = arg
        # 行数， 列数
        h, w = 482, 512
        # 裁剪图片的步长
        self.stride = stride
        # 每行每列裁剪的块数
        self.patch_per_row = (w - patch_size) // stride + 1
        self.patch_per_col = (h - patch_size) // stride + 1
        # 每张图像的裁剪块总数
        self.patch_per_img = self.patch_per_row * self.patch_per_col

        # 设置hyperspectral img和RGB img数据的路径
        self.hyer_root = '/root/autodl-tmp/Train_spectral/'
        self.rgb_root = '/root/autodl-tmp/Train_RGB/'

        # 从文件读取训练图像的路径列表
        with open(f'/root/code/dataset_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]
        hyper_list.sort()  # 对列表进行排序
        bgr_list.sort()

        """        
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        
        # len(hyper) of ntire2022 dataset:900
        # len(bgr) of ntire2022 dataset:900
        """
        for i in range(len(hyper_list)):
            h5py_file = 0
            sio_file = 0
            hyper_path = self.hyer_root + hyper_list[i]

            if hyper_path.endswith('.mat'):
                # 如果是MATLAB v7.3文件，改用h5py
                mat_data = h5py.File(hyper_path, 'r')
                hyper = np.float32(np.array(mat_data['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])

            bgr_path = self.rgb_root + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)  # 使用OpenCV读取RGB图像
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # 如果需要，将BGR格式转换为RGB格式
            bgr = np.float32(bgr)  # 将图像转换为float32格式
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())  # 归一化图像数据到[0, 1]
            bgr = np.transpose(bgr, [2, 0, 1])  # 调整RGB图像的维度 [H, W, C] -> [C, H, W]

            # 将高光谱和RGB图像加入到列表中
            self.hypers.append(hyper)
            self.RGBs.append(bgr)
            mat_data.close()
            print(f'Ntire2022 scene {i} is loaded.')


        """
        print(len(self.hypers), len(self.RGBs))
        # 899 899
        """

        self.img_num = len(hyper_list)
        print(self.img_num)
        self.patch_total_num = self.patch_per_img * self.img_num

    def argument(self, img, rotTimes, vFlip, hFlip):
        # 数据增强函数，对输入图像进行随机旋转和翻转
        # 任意生成一个旋转和翻转的角度
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))  # 旋转90度
        # 随机垂直翻转
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
            # 按照步长为-1切片(反向索引)
        # 随机水平翻转
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        patch_size = self.patch_size

        # 计算图像索引和裁剪块索引
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_row, patch_idx % self.patch_per_row
        # patch_index / patch_per_row = patch_idx % self.patch_per_col
        bgr = self.RGBs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:, h_idx * stride:h_idx * stride + patch_size, w_idx * stride:w_idx * stride + patch_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + patch_size, w_idx * stride:w_idx * stride + patch_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.argument(bgr, rotTimes, vFlip, hFlip)
            hyper = self.argument(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)
        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

    def __len__(self):
        return self.patch_total_num
class ValidDataset(Dataset):
    def __init__(self, bgr2rgb=True):
        self.hypers = []
        self.RGBs = []
        self.hyer_root = '/root/autodl-tmp/Valid_spectral/'
        self.rgb_root = '/root/autodl-tmp/Valid_RGB/'
        with open(f'./dataset_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        for i in range(len(hyper_list)):
            hyper_path = self.hyer_root + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = self.rgb_root + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.RGBs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')
        """
        print(len(self.RGBs), len(self.hypers))
        # 50 50
        """

    def __getitem__(self, idx):
        #整张进行测试
        hyper = self.hypers[idx]
        bgr = self.RGBs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)


# if __name__ == '__main__':
#     # 创建数据集实例并展示数据
#     train_dataset = TrainDataset(patch_size=128)
#     valid_dataset = ValidDataset()

