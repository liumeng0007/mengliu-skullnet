import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np
import datetime
from tqdm import tqdm

from model3d import TransNet3d
from fitnet import fit_net
from data import sex_ary, get_pixel

from sklearn.model_selection import KFold


def get_imgs():
    with open("imgpath.txt", "r") as f:
        allimgs = f.readlines()
        f.close()
    image3d_path = []
    for item in allimgs:
        item = item.split("\n")[0]
        # print(item)
        image3d_path.append(item)
    return image3d_path

image3d_path = get_imgs()
print(len(image3d_path))  # 1085
# label_ary = np.array(sex_ary, dtype=np.float32)
label_ary = torch.from_numpy(sex_ary).long()

class SkullDs(Dataset):
    def __init__(self, image_list, label_ary):
        self.image_list = image_list
        self.label_list = label_ary
        # self.transform = transformer

    def __getitem__(self, item):
        image = self.image_list[item]
        label = self.label_list[item]

        image3d = get_pixel(image)

        image3d = self.transform(image3d)
        image3d = np.expand_dims(image3d, 0)  # (1, 224, 224, 224)
        # label = self.transform(label)
        print(image)

        return image3d, label

    def __len__(self):
        return len(self.label_list)


def main():
    model = TransNet3d()
    # checkpoint = torch.load("model.pth")
    # model = model.load_state_dict(checkpoint["model"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.0005)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    EPOCH = 200
    max_acc = 0.9
    k = 10
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    imgs = image3d_path
    labels = label_ary
    data = SkullDs(imgs, labels)  # 所有的图像和标签都打包成了dataset

    fold = 1
    for train_index, val_index in kf.split(data):
        print(f"fold:{fold}")
        train_ds = dataset.Subset(data, train_index)
        val_ds = dataset.Subset(data, val_index)
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
            print(x.shape)

        fold_acc_train = 0
        fold_acc_val = 0
        for epoch in tqdm(range(EPOCH)):
            train_loss, train_acc, test_loss, test_acc = fit_net(model,
                                                                 train_dl,
                                                                 val_dl,
                                                                 loss_fn,
                                                                 optimizer,
                                                                 exp_lr_scheduler)
            fold_acc_train += train_acc
            fold_acc_val += test_acc
            # every epoch:
            output = f"fold:{fold},epoch:{epoch+1}/{EPOCH}, train_loss:{train_loss:.4f}, train_accurcy:{train_acc:.4f}, " \
                     f"test_loss:{test_loss:.4f}, test_accurcy:{test_acc:.4f}, " \
                     f"every_fold:{fold_acc_train/EPOCH, fold_acc_val/EPOCH}"
            with open("logger_6.txt", "a+") as f:
                f.write(output+"\n")

            if test_acc > max_acc:
                max_acc = test_acc
                checkpoint = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "fold": fold, "lr": exp_lr_scheduler.state_dict()
                }
                torch.save(model.state_dict(), "bestmodel.pth")

        fold += 1



if __name__ == '__main__':
    main()
