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

        # image3d = self.transform(image3d)
        image3d = np.expand_dims(image3d, 0)  # (1, 224, 224, 224)
        # label = self.transform(label)
        print(image)

        return image3d, label

    def __len__(self):
        return len(self.label_list)

######################
#  待重新定义

# train_imgs = image3d_path[:1000]
# train_label = label_ary[:1000]

# test_imgs = image3d_path[1000:]
# test_label = label_ary[1000:]
##############################
# 待重新定义

# train_ds = SkullDs(train_imgs, train_label)
# test_ds = SkullDs(test_imgs, test_label)

# train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
# test_dl = DataLoader(test_ds, batch_size=1)
##################################

# img, label = next(iter(train_dl))
# print(img.shape, label.shape)
# print(type(img), type(label))


# def get_k_fold(k, kf, data, model, loss_fn, optim, exp_lr_scheduler):
#     # k-fold cross validation
#
#     acc_train_mean = 0
#     loss_train_mean = 0
#     acc_val_mean = 0
#     loss_val_mean = 0
#
#     for train_index, val_index in kf.split(data):
#         train_ds = dataset.Subset(data, train_index)
#         val_ds = dataset.Subset(data, val_index)
#
#         train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
#         val_dl = DataLoader(val_ds, batch_size=1)
#
#         num_correct = 0
#         num_total = 0
#         running_loss = 0
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model = model.to(device)
#
#         model.train()
#         for x, y in tqdm(train_dl):
#             # x, y = torch.from_numpy(x), torch.from_numpy(y)
#             x = x.to(device)
#             y = y.to(device)
#             y_train_pred = model(x)
#             loss = loss_fn(y_train_pred, y)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             with torch.no_grad():
#                 y_train_pred = torch.argmax(y_train_pred, dim=1)  # 返回一个包含最大值索引位置的张量
#                 # print("y_train_pred:", y_train_pred)
#                 num_correct += (y_train_pred == y).sum().item()
#                 num_total += y.size(0)
#                 running_loss += loss.item()
#         exp_lr_scheduler.step()
#         train_acc = num_correct / num_total
#         print("train_acc:", train_acc)
#         train_loss = running_loss / len(train_dl.dataset)
#
#         acc_train_mean += train_acc
#         print("acc_train_acc:", acc_train_mean)
#         loss_train_mean += train_loss
#
#         num_correct_test = 0
#         num_total_test = 0
#         running_loss_test = 0
#         model.eval()
#         with torch.no_grad():
#             for x, y in tqdm(val_dl):
#                 x = x.to(device)
#                 y = y.to(device)
#                 y_test_pred = model(x)
#                 loss = loss_fn(y_test_pred, y)
#                 y_test_pred = torch.argmax(y_test_pred, dim=1)
#                 num_correct_test += (y_test_pred == y).sum().item()
#                 num_total_test += y.size(0)
#                 running_loss_test += loss.item()
#
#         test_acc = num_correct_test / num_total_test
#         print("test_acc:", test_acc)
#         test_loss = running_loss_test / len(val_dl.dataset)
#
#         acc_val_mean += test_acc
#         print("acc_val_mean:", acc_val_mean)
#         loss_val_mean += test_loss
#
#     return loss_train_mean/k, acc_train_mean/k, loss_val_mean/k, acc_val_mean/k


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
        for x, y in train_dl:
            # print(x)
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
                # checkpoint = {
                #     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                #     "fold": fold, "lr": exp_lr_scheduler.state_dict()
                # }
                torch.save(model.state_dict(), "bestmodel.pth")

        fold += 1


if __name__ == '__main__':
    main()
