import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import csv
import cv2

# 读取标签
def read_csv():
    csvfile = open('F:\Python\code\data\mappings.csv')
    reader = csv.reader(csvfile)
    lables = []
    for line in reader:
        tmpLine = [line[0], line[1]]
        # print(tmpLine)
        lables.append(tmpLine)
    csvfile.close()
    print(lables)
    return lables

# 读入图片
def read_img(lables):
    x = []
    y = []
    picnum = len(lables)
    print("picnum : ", picnum)
    for i in range(0, picnum):
        img_name = "F:\Python\code\data\data/" + lables[i][0] + '.jpg'
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        x.append(img)
        y.append(lables[i][1])
    return x, y

# 数据one—hot处理
def format_data(x, y):
    labeldict = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
        'K': 10,
        'L': 11,
        'M': 12,
        'N': 13,
        'O': 14,
        'P': 15,
        'Q': 16,
        'R': 17,
        'S': 18,
        'T': 19,
        'U': 20,
        'V': 21,
        'W': 22,
        'X': 23,
        'Y': 24,
        'Z': 25,
        '0': 26,
        '1': 27,
        '2': 28,
        '3': 29,
        '4': 30,
        '5': 31,
        '6': 32,
        '7': 33,
        '8': 34,
        '9': 35
    }

    tmp = []
    for i in range(len(y)):
        c0 = labeldict[y[i][0]]
        c1 = labeldict[y[i][1]]
        c2 = labeldict[y[i][2]]
        c3 = labeldict[y[i][3]]
        c4 = labeldict[y[i][4]]
        tmp.append(c0)
        tmp.append(c1)
        tmp.append(c2)
        tmp.append(c3)
        tmp.append(c4)

    x = np.array(x)
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, dim=1)
    x = x.type(torch.FloatTensor) / 255.
    print("-----------------------")
    number = 50000
    class_num = 36
    num_char = 5
    yt = torch.LongTensor(tmp)
    yt = torch.unsqueeze(yt, 1)
    # yt = torch.LongTensor(batch_size, 1).random_() % class_num
    yt_onehot = torch.FloatTensor(number, class_num)
    yt_onehot.zero_()
    yt_onehot.scatter_(1, yt, 1)
    yt_onehot = yt_onehot.view(-1, 180)
    y = yt_onehot
    print(y.size())
    print(x.size())
    print(torch.cuda.is_available())
    # 划分训练集和测试集
    train_x = x[:8000]
    train_y = y[:8000]
    test_x = x[8000:]
    test_y = y[8000:]
    return train_x, train_y, test_x, test_y

# 训练模型
class CNN(nn.Module):
    def __init__(self, num_class=36, num_char=5):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(

                nn.Conv2d(1, 64, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 128, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                )
        self.fc = nn.Linear(128*12*3, self.num_class*self.num_char)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*12*3)
        x = self.fc(x)
        return x

# 训练
def train(train_x, train_y):
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    batsize = 20
    epoches = 1
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    loss_func = nn.MultiLabelSoftMarginLoss()

    for epoch in range(epoches):
        losses = []
        iters = int(math.ceil(train_x.shape[0] / batsize))
        for i in range(iters):
            train_x_i = train_x[i * batsize: (i + 1) * batsize]
            train_y_i = train_y[i * batsize: (i + 1) * batsize]
            if torch.cuda.is_available():
                tx = train_x_i.cuda()
                ty = train_y_i.cuda()
            else:
                tx = Variable(train_x_i)
                ty = Variable(train_y_i)
            out = cnn(tx)
            loss = loss_func(out, ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        print('[%d/%d] Loss: %.5f' % (epoch + 1, epoches, loss.item()))
        torch.save(cnn.state_dict(), "./bin/cnn.pkl")

    # 测试
def test(test_x, lables):
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    outdict = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    batsize = 10
    with torch.no_grad():
        cnn.load_state_dict(torch.load('./bin/cnn.pkl'))
        iters = int(math.ceil(test_x.shape[0] / batsize))
        correct_sum = 0
        for epoch in range(iters):
            test_x_i = test_x[epoch * batsize: (epoch + 1) * batsize]
            if torch.cuda.is_available():
                tx_i = test_x_i.cuda()
            test_output = cnn(tx_i)
            correct_num = 0
            for i in range(test_output.size()[0]):
                c0 = outdict[np.argmax(test_output[i, 0:36].data.cpu().numpy())]
                c1 = outdict[np.argmax(test_output[i, 36:36 * 2].data.cpu().numpy())]
                c2 = outdict[np.argmax(test_output[i, 36 * 2:36 * 3].data.cpu().numpy())]
                c3 = outdict[np.argmax(test_output[i, 36 * 3:36 * 4].data.cpu().numpy())]
                c4 = outdict[np.argmax(test_output[i, 36 * 4:36 * 5].data.cpu().numpy())]
                c = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)
                if c == lables[8000 + 10 * epoch + i][1]:
                    correct_num += 1
                    print(c,lables[8000 + 10 * epoch + i][1])
            correct_sum += correct_num
            # print("Test accurate :", float(correct_num) / len(test_output))
        print("All Test accurate :", float(correct_sum) / len(test_x))