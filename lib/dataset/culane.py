import os
import numpy as np
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

__all__ = ["CULane"]


class CULane(data.Dataset):
    num_class = 5

    def __init__(self, data_root, is_train=True):
        super().__init__()
        self.data_root = data_root
        self.txt_path = os.path.join(data_root, "list", "{}_gt.txt".format("train" if is_train else "val"))
        self.indexes = self._create_index_lst()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229))  # image-net bgr
        ])

        self.is_train = is_train
        if is_train:
            from albumentations import Compose, ShiftScaleRotate, RandomBrightnessContrast, MotionBlur
            self.transforms = Compose([
                RandomBrightnessContrast(),
                MotionBlur(blur_limit=5, p=0.2),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT)
            ])

    def __getitem__(self, index):
        img_path, label_path, mask_code = self.indexes[index]
        img = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (800, 288), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (800, 288), interpolation=cv2.INTER_NEAREST)

        exist_code = np.array([int(c) for c in mask_code])
        if self.is_train:
            img, label, exist_code = self._transform(img, label, exist_code)

        debug = False
        if debug:
            show_label = cv2.applyColorMap(50 * label, cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.8, show_label, 0.2, 0)
            cv2.imshow('label', img)
            cv2.waitKey()

        img = self.to_tensor(img).float()
        label = torch.from_numpy(label).long()
        exist_code = torch.from_numpy(exist_code).float()

        return img, label, exist_code

    def __len__(self):
        return len(self.indexes)

    def _transform(self, img, label, exist_code):
        img, label, exist_code = self._random_flip(img, label, exist_code)
        augmented = self.transforms(image=img, mask=label)
        img, label = augmented['image'], augmented['mask']
        img = np.asarray(img).copy()
        label = np.asarray(label).copy()
        return img, label, exist_code

    @staticmethod
    def _random_flip(img, label, exist_code):
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]
            label = label[:, ::-1]
            label[label == 1] = 5
            label[label == 4] = 1
            label[label == 5] = 4
            label[label == 2] = 6
            label[label == 3] = 2
            label[label == 6] = 3
            exist_code = exist_code[::-1].copy()
            # label = label.copy()
        return img, label, exist_code

    def _create_index_lst(self):
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
        indexes = []
        for line in lines:
            img_path, label_path = line[:-9].split(' ')
            mask_code = line[-9:-1]
            image_path = os.path.join(self.data_root, img_path[1:])
            label_path = os.path.join(self.data_root, label_path[1:])
            indexes.append([image_path, label_path, mask_code.replace(' ', '')])
        return indexes


if __name__ == '__main__':
    txt_path = '/home/zns/dataset/lane/culane'
    dataset = CULane(txt_path, is_train=False)
    print(len(dataset))
