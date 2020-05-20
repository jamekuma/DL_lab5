import torch
from torch import nn
from torch.utils.data import Dataset
from MyDataset import MyDataset
from GAN import Discriminator, Generator
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--noise_size', type=int, default=2)
    parser.add_argument('--GM', type=int, default=384, help='middle size of Generator')
    parser.add_argument('--GO', type=int, default=2, help='out size of Generator')
    parser.add_argument('--DM', type=int, default=384, help='middle size of Discriminator')
    return parser.parse_args()


def visualize(G, D, real_data, epoch, save_path):
    """
    用于可视化的函数
    """
    # 关闭训练模式
    G.eval()
    D.eval()

    # 随机生成噪声
    noise = torch.randn(1000, args.noise_size).to(device)

    # Generator生成数据
    fake = G(noise)
    fake_np = fake.cpu().detach().numpy()

    # 画图需要用到的原始数据和生成数据的最大最小范围
    x_low, x_high = min(np.min(fake_np[:, 0]), -0.5), max(np.max(fake_np[:, 0]), 1.5)
    y_low, y_high = min(np.min(fake_np[:, 1]), 0), max(np.max(fake_np[:, 1]), 1)

    # 采样
    a_x = np.linspace(x_low, x_high, 200)
    a_y = np.linspace(y_low, y_high, 200)
    u = [[x, y] for y in a_y[::-1] for x in a_x[::-1]]
    u = np.array(u)
    u2tensor = torch.FloatTensor(u).cuda().to(device)

    # 判别器计算
    out = D(u2tensor)
    out2np = out.cpu().detach().numpy()

    # 绘制判别器的结果(黑白热度图)，存储在设定的文件路径中
    plt.cla()
    plt.clf()
    plt.axis('off')
    disc_path = os.path.join(save_path, 'Discriminator')
    if not os.path.exists(disc_path):
        os.makedirs(disc_path)
    plt.imshow(out2np.reshape(200, 200), extent=[x_low, x_high, y_low, y_high], cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(disc_path, 'epoch{}.png'.format(epoch)))

    # 绘制生成器的结果，存储在设定的文件路径中
    plt.cla()
    plt.clf()
    plt.axis('off')
    plt.imshow(out2np.reshape(200, 200), extent=[x_low, x_high, y_low, y_high], cmap='gray')
    plt.colorbar()
    plt.scatter(real_data[:, 0], real_data[:, 1], c='b', s=10)
    plt.scatter(fake_np[:, 0], fake_np[:, 1], c='r', s=10)
    g_path = os.path.join(save_path, 'Generator')
    if not os.path.exists(g_path):
        os.makedirs(g_path)
    plt.savefig(os.path.join(g_path, 'epoch{}.png'.format(epoch)))


def train(type, dataset:Dataset):
    train_set = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    G = Generator(args.noise_size, args.GM, args.GO).to(device)
    D = Discriminator(args.GO, args.DM, 'gan').to(device)

    # optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_G = torch.optim.SGD(G.parameters(), lr=args.lr)
    # optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    optimizer_D = torch.optim.SGD(D.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        G.train()
        D.train()
        loss_G_avg = 0.0
        loss_D_avg = 0.0
        for real_data in train_set:
            
            
            # 更新D
            optimizer_D.zero_grad()
            real_data = real_data.to(device)  # 真实的数据
            noise = torch.randn(real_data.size(0), args.noise_size).to(device)   # 随机噪声
            fake_data = G(noise).to(device)     # 生成的数据（假数据）
            # log(D(x)+log(1-D(G(z))))  注意fake_data这里不参加backward故detach
            loss_D = -(torch.log(D(real_data)) + torch.log(torch.ones(args.batch_size).to(device) - D(fake_data.detach()))).mean()
            loss_D.backward()
            optimizer_D.step()
            loss_D_avg += loss_D.item()

            # 更新G
            optimizer_G.zero_grad()
            noise = torch.randn(real_data.size(0), args.noise_size).to(device)  # 随机噪声
            fake_data = G(noise).to(device)  # 生成的数据（假数据）
            loss_G = (torch.log(torch.ones(args.batch_size).to(device) - D(fake_data))).mean() # log(1-D(G(z))))
            loss_G.backward()
            optimizer_G.step()
            loss_G_avg += loss_G.item()
        loss_G_avg /= len(train_set)
        loss_D_avg /= len(train_set)
        print('Epoch  {}  loss_G: {:.6f}  loss_D: {:.6f}'.format(epoch + 1, loss_G_avg, loss_D_avg))
        if epoch % 5 == 0:
            visualize(G, D, dataset.get_numpy_data(), epoch + 1, type)




args = get_args()
dataset = MyDataset('./points.mat')
train('gan', dataset)


