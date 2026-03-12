import torch
import h5py
import os
import gc
import numpy as np
import torch.utils.data as Data
from models import MFSTNet
import torch.nn.parallel
import torch.nn as nn
from ConfusionMatrix import ConfusionMatrix
import random
from sklearn.metrics import f1_score
import scipy.io as sio
from thop import profile
import time

def input_cat(input_mat_name, input_tem_name, split_ratio):

    input_1 = sio.loadmat(input_mat_name)
    input_1 = input_1['input_pic'][()]
    sample_Shuffle = np.load('/media/MFSTNet/sample_Shuffle.npy')
    input_1 = input_1[sample_Shuffle]

    input_1_train = input_1[:int(input_1.shape[0] * split_ratio)][:][:][:]
    input_1_test = input_1[int(input_1.shape[0] * split_ratio):][:][:][:]
    del input_1
    gc.collect()

    input_1_train = torch.from_numpy(input_1_train).type(torch.float32) / 255.0
    input_1_test = torch.from_numpy(input_1_test).type(torch.float32) / 255.0

    input_2 = torch.load(input_tem_name)
    indices = torch.tensor(sample_Shuffle, dtype=torch.long)
    input_2 = input_2[indices]
    input_2_train = input_2[:int(len(input_2) * split_ratio)]
    input_2_test = input_2[int(len(input_2) * split_ratio):]
    del input_2
    gc.collect()

    return input_1_train, input_1_test, input_2_train, input_2_test


def data_pre(input_mat_name, input_tem_name, output_mat_name, BATCH_SIZE, split_ratio):

    label = h5py.File(name=output_mat_name, mode='r')
    label = label['label'][()]
    label = label.squeeze()
    sample_Shuffle = np.load('/media/MFSTNet/sample_Shuffle.npy')
    label = label[sample_Shuffle]

    label = torch.from_numpy(label).type(torch.long)
    label_train = label[:int(label.shape[0] * split_ratio)]
    label_test = label[int(label.shape[0] * split_ratio):]
    print('label_train 0/1/2: {}/{}/{}'.format(
        (label_train == 0).sum(), (label_train == 1).sum(), (label_train == 2).sum()))
    print('label_test 0/1/2: {}/{}/{}'.format(
        (label_test == 0).sum(), (label_test == 1).sum(), (label_test == 2).sum()))

    class_sample_count = torch.tensor(
        [(label == t).sum() for t in torch.unique(label, sorted=True)])
    weight1 = 1 - class_sample_count/class_sample_count.sum()

    del label
    gc.collect()

    input_1_train, input_1_test, input_2_train, input_2_test = input_cat(input_mat_name, input_tem_name, split_ratio)
    gc.collect()

    # 制成样本
    class FuseDataset(Data.Dataset):
        def __init__(self, input_1, input_2, labels):

            self.input_1 = input_1
            self.input_2 = input_2
            self.label = labels

        def __getitem__(self, index):
            return self.input_1[index], self.input_2[index], self.label[index]

        def __len__(self):
            return self.input_1.shape[0]

        def get_labels(self):
            return self.label

    fuse_train_Dataset = FuseDataset(
        input_1_train,
        input_2_train,
        label_train
    )
    fuse_test_Dataset = FuseDataset(
        input_1_test,
        input_2_test,
        label_test
    )

    train_loader = Data.DataLoader(
        fuse_train_Dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )
    test_loader = Data.DataLoader(
        fuse_test_Dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        pin_memory=True
    )

    print('data_pre new done!')

    return train_loader, test_loader

def train(train_loader, test_loader):
    EPOCH = 100
    LR = 0.0001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MFSTNet()
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_cross = nn.CrossEntropyLoss()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(BASE_DIR, 'MFST.pkl')

    train_loss_plot = []
    train_acc_plot = []
    test_acc_plot = []
    test_loss_plot = []
    best_acc = 0
    for epoch in range(EPOCH):

        if epoch == 25 or epoch == 50 or epoch == 75 or epoch == 100:
            LR = LR * 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        model.train()
        running_loss = 0.0
        train_acc = 0.0
        for step, (input_1_train, input_2_train, label_source) in enumerate(train_loader):
            input_1_train = input_1_train.to(device)
            input_2_train = input_2_train.to(device)
            label_source_pred = model(input_1_train, input_2_train)
            label_source_pred = label_source_pred.squeeze(-1)
            predict_source = torch.max(label_source_pred.to(device), dim=1)[1]
            train_acc += (predict_source.to(device) == label_source.to(device)).sum().item()
            loss_cls = loss_cross(label_source_pred, label_source.to(device)).to(device)
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            running_loss += loss_cls.item()

        train_accuracy = train_acc / (len(train_loader) * batch_size)  # train_loader*batch_size)
        running_loss /= len(train_loader)
        train_loss_plot.append(running_loss)
        train_acc_plot.append(train_accuracy)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for input_1_test, input_2_test, target in test_loader:
                input_1_test = input_1_test.to(device)
                input_2_test = input_2_test.to(device)
                target = target.to(device)
                pred1 = model(input_1_test, input_2_test)
                test_loss += loss_cross(pred1, target.to(device)).item()
                pred1 = pred1.squeeze(-1)
                pred1 = pred1.data.max(1)[1]
                # correct += pred1.eq(target.data.view_as(pred1)).cpu().sum()
                correct += (pred1.to(device) == target.to(device)).sum().item()

            test_loss /= len(test_loader)
            test_accurate = correct / (len(test_loader) * batch_size)
            test_loss_plot.append(test_loss)
            test_acc_plot.append(test_accurate)

            print('[epoch %d] train_loss: %.4f  train_accuracy: %.4f  test_loss: %.4f  test_accuracy: %.4f' % (epoch + 1, running_loss, train_accuracy, test_loss,  test_accurate))

        if test_accurate > best_acc:
            best_acc = test_accurate
            torch.save(model.state_dict(), save_path)


def predict(test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MFSTNet()

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    load_path = "./weights/a.pkl"
    model.load_state_dict(torch.load(load_path))

    model.eval()

    confusion = ConfusionMatrix(num_classes=3, labels=['0', '1', '2'])
    acc = 0.0
    show_output = []
    real_label = []
    features = []
    labels = []

    total_time = 0
    total_samples = 0

    with torch.no_grad():
        for input_1_test, input_2_test, target in test_loader:

            input_1_test = input_1_test.to(device)
            input_2_test = input_2_test.to(device)
            target = target.to(device)

            batch_size = input_1_test.size(0)
            torch.cuda.synchronize()
            start_time = time.time()

            test_output = model(input_1_test, input_2_test)

            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            total_samples += batch_size

            labels.append(target)
            features.append(test_output)
            test_output = test_output.squeeze(-1)
            predict_y = torch.max(test_output, dim=1)[1]
            show_output.extend(predict_y.cpu().numpy())
            acc += (predict_y == target).sum().item()
            real_label.extend(target.cpu().numpy())
            confusion.update(predict_y.to('cpu').numpy(), target.to('cpu').numpy())

        val_accurate = acc / len(real_label)

        f1 = f1_score(real_label, show_output, average='macro')

        avg_time = total_time / total_samples
        fps = 1 / avg_time
        print("\n===== Inference Speed =====")
        print("Average inference time: %.4f s/sample" % avg_time)
        print("FPS: %.2f" % fps)

        print(f'Test F1 Score: {f1:.4f}')
        print('val_accurate=', val_accurate * 100.0, '%')
        print('len(val):', len(real_label), '\n', 'len(acc):', acc)

        confusion.summary()
        confusion.plot()

        return val_accurate


def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':


    seed_torch()
    split_ratio = 0.8  # 分割率
    batch_size = 32
    size_4 = 1453

    input_mat_name = "/media/MFSTNet/final/input_112.mat"
    input_tem_name = "/media/MFSTNet/final/save_gaf.pth"
    output_mat_name = "/media/MFSTNet/final/label.mat"

    train_loader, test_loader = data_pre(input_mat_name, input_tem_name, output_mat_name, batch_size, split_ratio)

    for i, (x, z, y) in enumerate(test_loader):
        print("batch index {}, 0/1/2: {}/{}/{}".format(i, (y == 0).sum(), (y == 1).sum(), (y == 2).sum()))

    train(train_loader, test_loader)

    val_acc = []
    for counter in range(5):
        val_accurate = predict(test_loader)
        val_acc.append(val_accurate)

    print('val_acc:\n', val_acc)
    print('\ndone!')


