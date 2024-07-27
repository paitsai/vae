import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 定义Pokemon数据集
class Pokemon(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Pokemon, self).__init__()
        self.root = root
        self.image_path = [os.path.join(root, x) for x in os.listdir(root)]
        random.shuffle(self.image_path)

        if transform is not None:
            self.transform = transform

        if train:
            self.images = self.image_path[: int(.8 * len(self.image_path))]
        else:
            self.images = self.image_path[int(.8 * len(self.image_path)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item])


latent_dim = 32
inter_dim = 128
mid_dim = (256, 2, 2)
mid_num = 1
for i in mid_dim:
    mid_num *= i # 256*2*2=1024


class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        super(ConvVAE, self).__init__()

        self.encoder = nn.Sequential(

            # conv2d的四个参数分别为input_channel,output_channel,kernel_size,stride和padding
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32), # 第一个参数是特征数也就是通道数
            nn.LeakyReLU(.2),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2),
        )

        self.fc1 = nn.Linear(mid_num, inter_dim)
        self.fc2 = nn.Linear(inter_dim, latent * 2)

        self.fcr2 = nn.Linear(latent, inter_dim)
        self.fcr1 = nn.Linear(inter_dim, mid_num)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(16, 3, 4, 2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        # 进行随机采样
        epsilon = torch.randn_like(mu) 
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x = self.encoder(x)
        x = self.fc1(x.view(batch, -1))
        h = self.fc2(x)

        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        decode = self.fcr2(z)
        decode = self.fcr1(decode)
        recon_x = self.decoder(decode.view(batch, *mid_dim))

        return recon_x, mu, logvar


kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)


epochs = 2000
batch_size = 512

best_loss = 1e9
best_epoch = 0

valid_losses = []
train_losses = []

transform = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.ToTensor(),
])

pokemon_train = Pokemon('./pokemon/figure/', train=True, transform=transform)
pokemon_valid = Pokemon('./pokemon/figure/', train=False, transform=transform)

train_loader = DataLoader(pokemon_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(pokemon_valid, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvVAE()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0.
    train_num = len(train_loader.dataset)

    for idx, x in enumerate(train_loader):
        batch = x.size(0)
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        recon = recon_loss(recon_x, x)
        kl = kl_loss(mu, logvar)

        loss = recon + kl
        train_loss += loss.item()
        loss = loss / batch

        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 参数step更新

        if idx % 100 == 0:
            print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} in Step {idx}")

    train_losses.append(train_loss / train_num)

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.
    valid_num = len(test_loader.dataset)
    model.eval()
    with torch.no_grad():
        for idx, x in enumerate(test_loader):
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)
            loss = recon + kl
            valid_loss += loss.item()
            valid_kl += kl.item()
            valid_recon += recon.item()

        valid_losses.append(valid_loss / valid_num)

        print(
            f"Valid loss {valid_loss / valid_num: .3f} \t Recon {valid_recon / valid_num: .3f} \t KL {valid_kl / valid_num: .3f} in epoch {epoch}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'best_model_pokemon')
            print("Model saved")



import numpy as np
from scipy.stats import norm


state = torch.load('best_model_pokemon')
model = ConvVAE()
model.load_state_dict(state)

n = 10
image_size = 40

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

model.eval()

selected = 21
coll = [(selected, i) for i in range(latent_dim) if i != selected]

for idx, (p, q) in enumerate(coll):
    figure = np.zeros((3, image_size * n, image_size * n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            t = [random.random() for i in range(latent_dim)]
            t[p], t[q] = xi, yi
            z_sampled = torch.FloatTensor(t).unsqueeze(0)
            with torch.no_grad():
                decode = model.fcr1(model.fcr2(z_sampled))
                decode = decode.view(1, *mid_dim)
                decode = model.decoder(decode)
                decode = decode.squeeze(0)

                figure[:,
                i * image_size: (i + 1) * image_size,
                j * image_size: (j + 1) * image_size
                ] = decode

plt.title("X: {}, Y: {}".format(p, q))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.imshow(figure.transpose(1, 2, 0))
plt.savefig("poke-vae.png")