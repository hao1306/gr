from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import glob
import cv2
import random
import argparse
import torch
import sys
sys.path.append('/mnt/ssd/hao/')
from utils import preprocess, preprocess_all

class ResNetDataset(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 使用当前帧和之前的input_n帧作为时序图像输入
        # predict_n: 预测未来路径点个数
        
        self.phase = phase
        self.cfg = cfg

        data_path = cfg.data_path + phase + '/'

        self.data_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        self.crop_ij_max = [self.image_size[0] - self.crop_size[0], self.image_size[1] - self.crop_size[1]]  # 
        self.crop_ij_val = [int((self.image_size[0] - self.crop_size[0])/2), int((self.image_size[1] - self.crop_size[1])/2)]

        filepaths = glob.glob(data_path + '*.csv')
        scene_n = len(filepaths)
        sample_n = np.zeros(scene_n, np.int32)

        f = open(filepaths[0], 'r')
        self.imagepaths = f.readlines()

        sample_n[0] = len(self.imagepaths)

        for i, filepath in enumerate(filepaths[1:]):
            f = open(filepath, 'r')
            self.imagepaths = self.imagepaths + f.readlines()

            sample_n[i + 1] = len(self.imagepaths)    
        
        self.sample_index = []

        if cfg.image_n == 1:
            self.imagepaths = self.imagepaths[::6]
            sample_n = (sample_n/6).astype(np.int32) # calculate number of sample, each sample has six pics
            for i in range(scene_n):
                if i == 0:
                    self.sample_index = [j for j in range(cfg.input_n - 1, sample_n[i] - cfg.predict_n)]
                else:
                    self.sample_index = self.sample_index + [j for j in range(sample_n[i-1] + cfg.input_n - 1, sample_n[i] - cfg.predict_n)]
        elif cfg.image_n == 6:
            sample_n = sample_n.astype(np.int32)
            for i in range(scene_n):
                if i == 0:
                    self.sample_index = [j for j in range((cfg.input_n - 1) * 6, sample_n[i] - cfg.predict_n * 6, 6)]
                else:
                    self.sample_index = self.sample_index + [j for j in range(sample_n[i-1] + (cfg.input_n - 1) * 6, sample_n[i] - cfg.predict_n * 6, 6)]

        
    def __len__(self):
        return len(self.sample_index)
    
    def img_process(self, imgpath):
        image = cv2.imread(self.data_root + imgpath)
        image = cv2.resize(image, self.image_size)
        if self.phase == 'train' or self.phase == 'mini_train':
                i = random.randint(0, self.crop_ij_max[0])
                j = random.randint(0, self.crop_ij_max[1])
                image = image[j:j+self.crop_size[1], i:i+self.crop_size[0]]
        else:
            image = image[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]


        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = image.astype(np.float32)/255.0
        image = torch.from_numpy(image)
        return image
    
    def img_process6(self, imgpath):
        img = cv2.imread(self.data_root + imgpath)
        img = preprocess_all(img)
        img = img.astype(np.float32)/255.0
        return img

        # image = cv2.imread(self.data_root + imgpath)
        # image = cv2.resize(image, self.image_size)
        # if self.phase == 'train' or self.phase == 'mini_train':
        #         i = random.randint(0, self.crop_ij_max[0])
        #         j = random.randint(0, self.crop_ij_max[1])
        #         image = image[j:j+self.crop_size[1], i:i+self.crop_size[0]]
        # else:
        #     image = image[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]


        # image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # image = image.astype(np.float32)/255.0
        # return image
    
    def resize_combined_img(self, image):
        img = cv2.resize(image, (450, 800))
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def __getitem__(self, idx):
        cfg = self.cfg

        coordinate_input = np.zeros((cfg.input_n, 2))
        coordinate_label = np.zeros((cfg.predict_n, 2))
        sample_token = None
        frame_idx = self.sample_index[idx]

        if cfg.image_n == 1:
            imagepath = []
            for i in range(cfg.input_n + cfg.predict_n):
                imgpath, x, y, token = self.imagepaths[frame_idx - cfg.input_n + i + 1].split(',')
                if i < cfg.input_n:
                    imagepath.append(imgpath)
                    coordinate_input[i, 0] = float(x)
                    coordinate_input[i, 1] = float(y)
                    if i == cfg.input_n - 1:
                        sample_token = str(token)
                else:
                    coordinate_label[i - cfg.input_n, 0] = float(x)
                    coordinate_label[i - cfg.input_n, 1] = float(y)

            minus_a = coordinate_input[3, 0]
            minus_b = coordinate_input[3, 1]
            coordinate_input[:, 0] = coordinate_input[:, 0] - minus_a
            coordinate_input[:, 1] = coordinate_input[:, 1] - minus_b
            coordinate_label[:, 0] = coordinate_label[:, 0] - minus_a
            coordinate_label[:, 1] = coordinate_label[:, 1] - minus_b
            # coordinate_input = np.resize(coordinate_input, (cfg.input_n * 2))
            # coordinate_label = np.resize(coordinate_label, (cfg.predict_n * 2))

            image = [self.img_process(path_) for path_ in imagepath]

            # print('len of image', len(image))
            # print(type(image))

            batch = torch.stack(image)

            xy_in = torch.from_numpy(np.array(coordinate_input).astype(np.float32))
            xy_label = torch.from_numpy(np.array(coordinate_label).astype(np.float32))
            # print(xy_label.shape)

        if cfg.image_n == 6:
            frame_idx = self.sample_index[idx]
            imagepath = []
            for i in range((cfg.input_n + cfg.predict_n) * cfg.image_n):
                imgpath, x, y, token = self.imagepaths[frame_idx - (cfg.input_n - 1) * 6 + i].split(',')
                        
                if i < cfg.input_n * 6:
                    imagepath.append(imgpath)
                    if i % 6 == 0:
                        t = i // 6
                        coordinate_input[t, 0] = float(x)
                        coordinate_input[t, 1] = float(y)
                    if i == (cfg.input_n - 1) * 6:
                        sample_token = str(token)
                else:
                    if i % 6 == 0:
                        t = i // 6
                        coordinate_label[t - cfg.input_n, 0] = float(x)
                        coordinate_label[t - cfg.input_n, 1] = float(y)

            # print('before minus', coordinate_label)
            minus_a = coordinate_input[3, 0]
            minus_b = coordinate_input[3, 1]
            coordinate_input[:, 0] = coordinate_input[:, 0] - minus_a
            coordinate_input[:, 1] = coordinate_input[:, 1] - minus_b
            coordinate_label[:, 0] = coordinate_label[:, 0] - minus_a
            coordinate_label[:, 1] = coordinate_label[:, 1] - minus_b
            # coordinate_input = np.resize(coordinate_input, (cfg.input_n * 2))
            # coordinate_label = np.resize(coordinate_label, (cfg.predict_n * 2))
            xy_in = torch.from_numpy(np.array(coordinate_input).astype(np.float32))
            xy_label = torch.from_numpy(np.array(coordinate_label).astype(np.float32))
            # print(coordinate_input.shape)

            image = [self.img_process6(path_) for path_ in imagepath]
            img = []
            for m in range(4):
                combined_image = np.zeros((2700, 3200, 3))
                for n in range(6):
                    row = n // 2
                    col = n % 2
                    combined_image[(row * 900) : ((row + 1) * 900), (col * 1600) : ((col + 1) * 1600), :] = image[m * 6 + n]
                img.append(combined_image)

            imgs = [self.resize_combined_img(img_) for img_ in img]

            batch = torch.stack(imgs)

        # print('xy_in size', xy_in.shape, 'xy label size', xy_label.shape)
        return batch, xy_in, xy_label, sample_token
    
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--data-root', type=str, default='dataset/nuscenes/trian/', help='path to save checkpoint')
    parser.add_argument('--data-path', type=str, default='dataset/nuscenes/image_scenes/', help='path to save checkpoint')
    parser.add_argument('--image-n', type=int, default=1, help='num of cameras to be used')
    parser.add_argument('--input-n', type=int, default=4, help='num of images as input')
    parser.add_argument('--predict-n', type=int, default=6, help='num of points to be predicted')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='path to save checkpoint')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='path to save checkpoint')

    return parser.parse_args()
    
if __name__=='__main__':
    cfg = parse_cfg()
    train_data = ResNetDataset(cfg, phase='val')
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    i = 0
    for (image, xy_in, target, token) in train_dataloader:
        i += 1
        # print('xy_in', xy_in)
        print('target', target)

        if i > 3:
            break
