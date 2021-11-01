
'''
@ author of the train.py:lry (Intelligent Awareness 2001)
@ train.py 作者：刘仁宇 智能感知2001
@ copyright (c) 2020-2021
@ all rights reserved
@ you must not use this python file in commercial use and commercial purpose
@ be careful :if you use this file, you should adjust the parameters by yourself
@ 注意：你应该在本文件中自己调整参数
@ train.py version 2.0.1
'''


import numpy as np
from PIL import Image
import cv2
import imageio
import os
import pickle
import re
import time
from PIL import Image
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}

g_frames = []
ans = 0


def make_raw_dataset(dataset='train'):
    data = []
    all_frames = []
    frames = []
    global ans
    global g_frames
    if dataset == 'train':
        for category in CATEGORIES:
            folder_path = os.path.join("..", "dataset", category)
            # print('folder_path=', folder_path)
            filenames = sorted(os.listdir(folder_path))
            # print(filenames)
            for filename in filenames:
                ans = ans+1
                filepath = os.path.join('..', 'dataset', category, filename)
                # print(filepath)
                vid = cv2.VideoCapture(filepath)
                frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = frame_count-2
                num = 0
                print('loading video please wait:,filepath=', filepath)
                while(vid.isOpened()):
                    ret, frame = vid.read()
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    except :
                        print("ERROR!!!,error filepath=", filepath)
                    if not ret:
                        break
                    cv2.imwrite('%s.jpg' % ('pic_' + str(num)), gray)
                    if num == frame_count:
                        break
                    num = num+1
                img_number = frame_count
                #print(img_number)
                all_img = [np.array(Image.open('pic_'+str(i)+'.jpg', 'r'))for i in range(img_number)]
                h = all_img[0].shape[0]       # height
                w = all_img[0].shape[1]       # weight
                # calculating to get the background photograph
                back_img = np.zeros((h, w))
                for single_img in all_img:
                    back_img += single_img
                back_img /= img_number        # in order to get the average of the photograph
                # save the background photograph
                Image.fromarray(back_img).convert('RGB')
                # 原视频与背景逐帧相减后取绝对值 得到前景
                front_img = np.array([i - back_img for i in all_img])
                front_img = front_img.__abs__()

                # 前景二值化 设定阈值将前景像素值化为0或1
                threshold_level = 80  # 阈值
                threshold = np.full((h, w), threshold_level)
                front_img = np.array([i < threshold for i in front_img], dtype=np.int8) * 255  # 逐帧与阈值比较 化为 元素为0或1的矩阵
                # 矩阵与同形矩阵的比较 小0 （大1）                                    #规定返回数据元素的数据类型

                # 在原帧上抠图 得到真实的前景
                front_img = np.fmax(np.array(front_img), all_img)

                # 保存
                for i in range(img_number):
                    Image.fromarray(front_img[i]).convert('RGB')
                print('begin to analyse the video:video filepath=', filepath)
                for i in range(img_number-1):
                    d = np.any(front_img[i] != 255)
                    if d:
                        frames.append(i)
                        all_frames.append([filepath, category, i])
                        g_frames.append([filepath, category, i])
                #print(frames)
                #print(all_frames)
            print('analyse category finished->category:', category)
        print('analyse all files ended')
        print('copy list from function to global,please wait')
        g_frames = all_frames
        g_frames = sorted(g_frames)
        all_frames = sorted(all_frames)
        print('copy finished')
        all_frames = all_frames[: int(0.8*ans), :, :]
        print(all_frames)
        print('begin to train raw dataset,please wait')
    elif dataset == 'dev':
        all_frames = g_frames
        all_frames = all_frames[int(0.8*ans): int(0.9*ans), :, :]
        print(all_frames)
    else:
        all_frames = g_frames
        all_frames = all_frames[int(0.9*ans):, :, :]
        print(all_frames)
    for category in CATEGORIES:
        # Get all files in current category's folder.
        folder_path = os.path.join("..", "dataset", category)
        filenames = sorted(os.listdir(folder_path))
        for filename in filenames:
            filepath = os.path.join("..", "dataset", category, filename)
            vid = imageio.get_reader(filepath, "ffmpeg")
            for i, frame in enumerate(vid):
                ok = False
                try:
                    if all_frames.index([filename, category, i]) >= 0:
                        ok = True
                    else:
                        ok = False
                except :
                    ok = False
                # Convert to grayscale.
                try:
                    frame = Image.fromarray(np.array(frame))
                    # print(frame)
                    frame = frame.convert("L")

                    frame = np.array(frame.getdata(),
                                            dtype=np.uint8).reshape((120, 160))
                    # frame = np.array(frame.getdata(),
                    #                  dtype=np.float32).reshape((120, 160))
                    # frame = imresize(frame, (60, 80))
                    frame = np.array(Image.fromarray(frame).resize((60, 80)))
                    frames.append(frame)
                except IndexError:
                    print('Index Error')
                    raise
        data.append({
            "filename": filename,
            "category": category,
            "frames": frames
        })
        print(g_frames)
        pickle.dump(data, open("data/%s.p" % dataset, "wb"))


def make_optflow_dataset(dataset='train'):
    global g_frames
    global ans
    all_frames = []
    if dataset == 'train':
        all_frames = g_frames
        all_frames = all_frames[:int(0.8*ans), :, :]
    elif dataset == 'dev':
        all_frames = g_frames
        all_frames = all_frames[int(0.8*ans):int(0.9*ans), :, :]
    else:
        all_frames = g_frames
        all_frames = all_frames[: int(0.9*ans), :, :]
    farneback_params = dict(
        winsize=20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

    # frames_idx = parse_sequence_file()

    data = []

    for category in CATEGORIES:
        # Get all files in current category's folder.
        folder_path = os.path.join("..", "dataset", category)
        filenames = sorted(os.listdir(folder_path))

    for filename in filenames:
        filepath = os.path.join("..", "dataset", category, filename)
        '''
        # Get id of person in this video.
        person_id = int(filename.split("_")[0][6:])
        if person_id not in ID:
            continue
        '''
        vid = imageio.get_reader(filepath, "ffmpeg")

        flow_x = []
        flow_y = []

        prev_frame = None
        for i, frame in enumerate(vid):
            ok = False
            try:
                if all_frames.index([filename, category, i]) >= 0:
                    ok = True
                    break
                else:
                    ok = False
                    continue
            except:
                ok = False
                continue
            frame = Image.fromarray(np.array(frame))
            frame = frame.convert("L")
            frame = np.array(frame.getdata(),
                             dtype=np.uint8).reshape((120, 160))
            # frame = imresize(frame, (60, 80))
            frame = np.array(Image.fromarray(frame).resize((60, 80)))

            if prev_frame is not None:
                # Calculate optical flow.
                flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                                     **farneback_params)
                subsampled_x = np.zeros((30, 40), dtype=np.float32)
                subsampled_y = np.zeros((30, 40), dtype=np.float32)

                # print(flows.shape)
                for r in range(30):
                    for c in range(40):
                        # subsampled_x[r, c] = flows[r*2, c*2, 0]
                        # subsampled_y[r, c] = flows[r*2, c*2, 1]

                        subsampled_x[r, c] = flows[c * 2, r * 2, 0]
                        subsampled_y[r, c] = flows[c * 2, r * 2, 1]

                flow_x.append(subsampled_x)
                flow_y.append(subsampled_y)

            prev_frame = frame

        data.append({
            "filename": filename,
            "category": category,
            "flow_x": flow_x,
            "flow_y": flow_y
        })

        pickle.dump(data, open("data/%s_flow.p" % dataset, "wb"))


def copyright():
    print('these information are about the copyright of this file:')
    print('\tauthor of the train.py:lry (Intelligent Awareness 2001)')
    print('\ttrain.py 作者：刘仁宇 智能感知2001')
    print('\tcopyright (c) 2020-2021')
    print('\tall rights reserved')
    print('\tyou must not use this python file in commercial use and commercial purpose')
    print('\tbe careful :if you use this file, you should adjust the parameters by yourself')
    print('\t注意：你应该在本文件中自己调整参数')
    print('\ttrain.py\tversion 2.0.1')


# the below code is about the net used in this file.
class CNNBlockFrameFlow(nn.Module):
    def __init__(self):
        super(CNNBlockFrameFlow, self).__init__()

        self.conv1_frame = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(4, 5, 5)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_frame = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_frame = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.conv1_flow_x = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_flow_x = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_flow_x = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.conv1_flow_y = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_flow_y = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_flow_y = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.fc1 = nn.Linear(3328, 128)
        self.dropfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, frames, flow_x, flow_y):
        out_frames = self.conv1_frame(frames)
        out_frames = self.conv2_frame(out_frames)
        out_frames = self.conv3_frame(out_frames)
        out_frames = out_frames.view(out_frames.size(0), -1)

        out_flow_x = self.conv1_flow_x(flow_x)
        out_flow_x = self.conv2_flow_x(out_flow_x)
        out_flow_x = self.conv3_flow_x(out_flow_x)
        out_flow_x = out_flow_x.view(out_flow_x.size(0), -1)

        out_flow_y = self.conv1_flow_y(flow_y)
        out_flow_y = self.conv2_flow_y(out_flow_y)
        out_flow_y = self.conv3_flow_y(out_flow_y)
        out_flow_y = out_flow_y.view(out_flow_y.size(0), -1)

        out = torch.cat([out_frames, out_flow_x, out_flow_y], 1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropfc1(out)
        out = self.fc2(out)

        return out

# net finished
# the code below is about how to train the data

def get_outputs(model, instances, flow=False, use_cuda=False):
    if flow:
        frames = Variable(instances["frames"])
        flow_x = Variable(instances["flow_x"])
        flow_y = Variable(instances["flow_y"])

        if use_cuda:
            frames = frames.cuda()
            flow_x = flow_x.cuda()
            flow_y = flow_y.cuda()

        outputs = model(frames, flow_x, flow_y)

    else:
        instances = Variable(instances)
        if use_cuda:
            instances = instances.cuda()

        outputs = model(instances)

    return outputs


def evaluate(model, dataloader, flow=False, use_cuda=False):
    loss = 0
    correct = 0
    total = 0

    # Switch to evaluation mode.
    model.eval()

    for i, samples in enumerate(dataloader):
        outputs = get_outputs(model, samples["instance"], flow=flow,
                              use_cuda=use_cuda)

        # labels = Variable(samples["label"])
        labels = Variable(samples["label"]).long()
        if use_cuda:
            labels = labels.cuda()

        # loss += nn.CrossEntropyLoss(size_average=False)(outputs, labels).data[0]
        loss += nn.CrossEntropyLoss(size_average=False)(outputs, labels).data

        score, predicted = torch.max(outputs, 1)
        correct += (labels.data == predicted.data).sum()

        total += labels.size(0)

    acc = correct / total
    loss /= total

    return loss, acc


def train(model, num_epochs, train_set, dev_set, lr=1e-3, batch_size=32,
          start_epoch=1, log=10, checkpoint_path=None, validate=True,
          resume=False, flow=False, use_cuda=False):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)

    # Must be sequential b/c this is used for evaluation.
    train_loader_sequential = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_set, batch_size=batch_size, shuffle=False)

    # Use Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Record loss + accuracy.
    hist = []

    # Check if we are resuming training from a previous checkpoint.
    if resume:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, "model_epoch%d.chkpt" % (start_epoch - 1)))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        hist = checkpoint["hist"]

    if use_cuda:
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Switch to train mode.
        model.train()

        for i, samples in enumerate(train_loader):

            # labels = Variable(samples["label"])
            labels = Variable(samples["label"]).long()
            if use_cuda:
                labels = labels.cuda()

            # Zero out gradient from previous iteration.
            optimizer.zero_grad()

            # Forward, backward, and optimize.
            outputs = get_outputs(model, samples["instance"], flow=flow,
                                  use_cuda=use_cuda)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % log == 0:
                # print("epoch %d/%d, iteration %d/%d, loss: %s"
                #       % (epoch, start_epoch + num_epochs - 1, i + 1,
                #       len(train_set) // batch_size, loss.data[0]))

                print("epoch %d/%d, iteration %d/%d, loss: %s"
                      % (epoch, start_epoch + num_epochs - 1, i + 1,
                         len(train_set) // batch_size, loss.data))

        # Get overall loss & accuracy on training set.
        train_loss, train_acc = evaluate(model, train_loader_sequential,
                                         flow=flow, use_cuda=use_cuda)

        if validate:
            # Get overall loss & accuracy on dev set.
            dev_loss, dev_acc = evaluate(model, dev_loader, flow=flow,
                                         use_cuda=use_cuda)

            print("epoch %d/%d, train_loss = %s, traic_acc = %s, "
                  "dev_loss = %s, dev_acc = %s"
                  % (epoch, start_epoch + num_epochs - 1,
                     train_loss, train_acc, dev_loss, dev_acc))

            hist.append({
                "train_loss": train_loss, "train_acc": train_acc,
                "dev_loss": dev_loss, "dev_acc": dev_acc
            })
        else:
            print("epoch %d/%d, train_loss = %s, train_acc = %s" % (epoch,
                                                                    start_epoch + num_epochs - 1, train_loss,
                                                                    train_acc))

            hist.append({
                "train_loss": train_loss, "train_acc": train_acc
            })

        optimizer.zero_grad()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hist": hist
        }

        # Save checkpoint.
        torch.save(checkpoint, os.path.join(
            checkpoint_path, "model_epoch%d.chkpt" % epoch))

# train finished
# the code below is about dataset

class RawDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = {
            "instance": self.instances[idx],
            "label": self.labels[idx]
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train"):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        for video in videos:
            for frame in video["frames"]:
                instances.append(frame.reshape((1, 60, 80)))
                labels.append(CATEGORY_INDEX[video["category"]])

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels

class BlockFrameDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = {
            "instance": self.instances[idx],
            "label": self.labels[idx]
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        for video in videos:
            for i, frame in enumerate(video["frames"]):
                current_block.append(frame)
                if len(current_block) % 15 == 0:
                    current_block = np.array(current_block)
                    instances.append(current_block.reshape((1, 15, 60, 80)))
                    labels.append(CATEGORY_INDEX[video["category"]])
                    current_block = []

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels

class BlockFrameFlowDataset(Dataset):
    def __init__(self, directory, dataset="train", mean=None):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        for i in range(len(self.instances)):
            self.instances[i]["frames"] = torch.from_numpy(
                self.instances[i]["frames"])
            self.instances[i]["flow_x"] = torch.from_numpy(
                self.instances[i]["flow_x"])
            self.instances[i]["flow_y"] = torch.from_numpy(
                self.instances[i]["flow_y"])

        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        sample = {
            "instance": self.instances[idx],
            "label": self.labels[idx]
        }

        return sample

    def zero_center(self, mean):
        for i in range(len(self.instances)):
            self.instances[i]["frames"] -= float(mean["frames"])
            self.instances[i]["flow_x"] -= float(mean["flow_x"])
            self.instances[i]["flow_y"] -= float(mean["flow_y"])

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            frame_path = os.path.join(directory, "train.p")
            flow_path = os.path.join(directory, "train_flow.p")
        elif dataset == "dev":
            frame_path = os.path.join(directory, "dev.p")
            flow_path = os.path.join(directory, "dev_flow.p")
        else:
            frame_path = os.path.join(directory, "test.p")
            flow_path = os.path.join(directory, "test_flow.p")

        video_frames = pickle.load(open(frame_path, "rb"))
        video_flows = pickle.load(open(flow_path, "rb"))

        instances = []
        labels = []

        mean_frames = 0
        mean_flow_x = 0
        mean_flow_y = 0

        for i_video in range(len(video_frames)):
            current_block_frame = []
            current_block_flow_x = []
            current_block_flow_y = []

            frames = video_frames[i_video]["frames"]
            flow_x = [0] + video_flows[i_video]["flow_x"]
            flow_y = [0] + video_flows[i_video]["flow_y"]

            for i_frame in range(len(frames)):
                current_block_frame.append(frames[i_frame])

                if i_frame % 15 > 0:
                    current_block_flow_x.append(flow_x[i_frame])
                    current_block_flow_y.append(flow_y[i_frame])

                if (i_frame + 1) % 15 == 0:
                    current_block_frame = np.array(
                        current_block_frame,
                        dtype=np.float32).reshape((1, 15, 60, 80))
                    current_block_flow_x = np.array(
                        current_block_flow_x,
                        dtype=np.float32).reshape((1, 14, 30, 40))
                    current_block_flow_y = np.array(
                        current_block_flow_y,
                        dtype=np.float32).reshape((1, 14, 30, 40))

                    mean_frames += np.mean(current_block_frame)
                    mean_flow_x += np.mean(current_block_flow_x)
                    mean_flow_y += np.mean(current_block_flow_y)

                    instances.append({
                        "frames": current_block_frame,
                        "flow_x": current_block_flow_x,
                        "flow_y": current_block_flow_y
                    })

                    labels.append(
                        CATEGORY_INDEX[video_frames[i_video]["category"]])

                    current_block_frame = []
                    current_block_flow_x = []
                    current_block_flow_y = []

        mean_frames /= len(instances)
        mean_flow_x /= len(instances)
        mean_flow_y /= len(instances)

        self.mean = {
            "frames": mean_frames,
            "flow_x": mean_flow_x,
            "flow_y": mean_flow_y
        }

        labels = np.array(labels, dtype=np.uint8)

        return instances, labels

# dataset finished
# the code below is about main function
def main():
    time0 = time.time()
    copyright()
    print('waiting for argparse...')
    parser = argparse.ArgumentParser(description="Block Frame&Flow ConvNet")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="directory to dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for training (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="number of epochs to train (default: 3)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="start index of epoch (default: 1)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for training (default: 0.001)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    start_epoch = args.start_epoch
    lr = args.lr
    log_interval = args.log

    if args.cuda == 1:
        cuda = True
        print('use cuda ...')
    else:
        cuda = False
        print('not use cuda ...')
    print('argparse setting finished')
    print("Making raw train dataset")
    make_raw_dataset(dataset="train")
    time1 = time.time()
    print("Making raw dev dataset")
    make_raw_dataset(dataset="dev")
    time2 = time.time()
    print("Making raw test dataset")
    make_raw_dataset(dataset="test")
    time3 = time.time()

    print("Making optical flow features for train dataset")
    make_optflow_dataset(dataset="train")
    time4 = time.time()
    print("Making optical flow features for dev dataset")
    make_optflow_dataset(dataset="dev")
    time5 = time.time()
    print("Making optical flow features for test dataset")
    make_optflow_dataset(dataset="test")
    time6 = time.time()
    print('print using time:')
    print('making raw data:time used->', time1 - time0, 's')
    print('making raw dev :time used->', time2 - time1, 's')
    print('making raw test:time used->', time3 - time2, 's')
    print('making optical flow features for train dataset:time used->', time4 - time3, 's')
    print('making optical flow features for dev dataset:time used->', time5 - time4, 's')
    print('making optical flow features for test dataset:time used->', time6 - time5, 's')
    print('making optical flow features succeed!(all dev,train,test/(both raw and non_raw))')

    print("Loading dataset")
    train_set = BlockFrameFlowDataset(dataset_dir, "train")
    dev_set = BlockFrameFlowDataset(dataset_dir, "dev")
    train_set.zero_center(train_set.mean)
    dev_set.zero_center(train_set.mean)

    # Create model and optimizer.
    model = CNNBlockFrameFlow()

    if start_epoch > 1:
        resume = True
    else:
        resume = False

    # Create directory for storing checkpoints.
    os.makedirs(os.path.join(dataset_dir, "cnn_block_frame_flow"),
                exist_ok=True)

    print("Start training")
    train(model, num_epochs, train_set, dev_set, lr=lr, batch_size=batch_size,
          start_epoch=start_epoch, log=log_interval,
          checkpoint_path=os.path.join(dataset_dir, "cnn_block_frame_flow"),
          validate=True, resume=resume, flow=True, use_cuda=cuda)

    print('training finished')


if __name__ == "__main__":
    main()
    print('successfully finished the process ')
    print('thanks for your usage')

# end
