import os
import cv2

import torch.utils.data as data
import torchvision.transforms as transforms

__all__ = ["CULaneTest"]


class CULaneTest(data.Dataset):
    num_class = 5

    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.txt_path = os.path.join(data_root, "list", "test.txt")
        self.indexes = self._create_index_lst()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229))  # image-net bgr
        ])

    def __getitem__(self, index):
        img_path = self.indexes[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 288), interpolation=cv2.INTER_NEAREST)

        img = self.to_tensor(img).float()

        return img, img_path

    def __len__(self):
        return len(self.indexes)

    def _create_index_lst(self):
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
        indexes = [os.path.join(self.data_root, line[1:-1]) for line in lines]
        return indexes


if __name__ == '__main__':
    txt_path = '/home/zns/dataset/lane/culane'
    dataset = CULaneTest(txt_path)
    print(len(dataset))
