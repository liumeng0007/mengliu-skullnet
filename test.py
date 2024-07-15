import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import SimpleITK as sitk
import torch
from model3d import TransNet3d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = "/home/omnisky/home/test"
path_label = "/home/omnisky/home/test/sex_test.txt"
l = os.listdir(path)
img_list = []
img_path = []

for item in l:
    if len(item) < 4:
        img_list.append(int(item))
img_list = sorted(img_list)

for i in img_list:
    img_path.append(os.path.join(path, f"{i}"))

sex_label = pd.read_csv(path_label, header=None)

def get_pixel_test(image):
    per = []
    per_dict = {}
    for item in glob.glob(os.path.join(image, "*")):
        # print(item)
        dcm = pydicom.dcmread(item)
        slope = dcm.RescaleSlope
        intercept = dcm.RescaleIntercept

        # get HU value: dcm * slope + intercept
        # InstanceNumber 排序
        per_dict[int(dcm.InstanceNumber)] = dcm.pixel_array * slope + intercept    # ct: -1000 - 3700

    for i in range(1, len(glob.glob(os.path.join(image, "*")))+1):
        pixel_ary = per_dict[i][50:470, 50:470]
        pixel_ary = np.copy(pixel_ary)
        pixel_ary_contig = np.ascontiguousarray(pixel_ary)

        # normlization and 0-mean
        max_value = np.max(pixel_ary_contig)
        min_value = np.min(pixel_ary_contig)
        pixel_ary_contig = (pixel_ary_contig - min_value) / (max_value - min_value)

        mean_value = np.mean(pixel_ary_contig)
        # print(mean_value)
        pixel_ary_contig -= mean_value

        pixel_ary_contig = np.resize(pixel_ary_contig, (224, 224))
        pixel_ary_contig = np.array(pixel_ary_contig, dtype=np.float32)

        per.append(pixel_ary_contig)

    person = np.stack(per, axis=0)[-224:, :, :]
    return person


image3d = get_pixel_test(img_path[0])
print(image3d.shape)


class SkullDs(Dataset):
    def __init__(self, image_list, label_ary):
        self.image_list = image_list
        self.label_list = label_ary
        # self.transform = transformer

    def __getitem__(self, item):
        image = self.image_list[item]
        label = self.label_list[item]

        image3d = get_pixel_test(image)

        # image3d = self.transform(image3d)
        image3d = np.expand_dims(image3d, 0)
        # label = self.transform(label)
        print(image)

        return image3d, label

    def __len__(self):
        return len(self.label_list)


def main():
    def load_model(model, model_save_path, use_state_dict=True):
        device = torch.device("cpu")  # 先加载到cpu
        if use_state_dict:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            model = torch.load(model_save_path, map_location=device)

        return model

    model = TransNet3d()
    model = load_model(model, "bestmodel.pth", use_state_dict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 放到gpu上

    test_dataset = SkullDs(img_path, sex_label)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    num_correct_test = 0
    num_total_test = 0
    running_loss_test = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_test_pred = model(x)
            loss = loss_fn(y_test_pred, y)
            y_test_pred = torch.argmax(y_test_pred, dim=1)
            num_correct_test += (y_test_pred == y).sum().item()
            num_total_test += y.size(0)
            running_loss_test += loss.item()

    test_acc = num_correct_test / num_total_test
    # print("test_acc:", test_acc)
    test_loss = running_loss_test / len(test_dataloader.dataset)

    output = f"test_acc:{test_acc}, test_loss:{test_loss}"
    with open("logger_test.txt", "a+") as f:
        f.write(output + "\n")

if __name__ == '__main__':
    main()











