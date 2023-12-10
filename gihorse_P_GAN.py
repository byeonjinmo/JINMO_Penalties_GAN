import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import time
import csv

# Hyper-parameters & Variables setting
num_epoch = 1000
batch_size = 64
learning_rate = 0.0002
img_size = 32
num_channel = 3  # CIFAR-10 has 3 channels
dir_name = "generator_images(b)"
noise_size = 150

# CSV 파일 준비
csv_file = os.path.join(dir_name, "generator_images(b).csv")
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Generator Loss", "Discriminator Loss", "Time Taken (seconds)", "Discriminator Accuracy"])

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))

# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize for RGB images

# CIFAR-10 dataset setting
CIFAR10_dataset = torchvision.datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=transform,
    download=True
)

# Data loader
data_loader = torch.utils.data.DataLoader(
    dataset=CIFAR10_dataset,
    batch_size=batch_size,
    shuffle=True
)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(num_channel, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # 가중치 초기화
        self._initialize_weights()

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

    # 허 초기화
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Declares generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(noise_size, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            nn.Conv2d(128, num_channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # 가중치 초기화
        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, noise_size)  # 변경된 부분
        out = self.l1(x)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # 허 초기화
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Initialize generator/Discriminator
discriminator = Discriminator().to(device)
generator = Generator().to(device)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Training part
for epoch in range(num_epoch):
    g_loss_accum = 0.0
    d_loss_accum = 0.0
    start_time = time.time()
    correct_real = 0
    correct_fake = 0
    total = 0

    for i, (images, _) in enumerate(data_loader):
        current_batch_size = images.size(0)

        # 현재 배치 크기에 맞는 실제 및 가짜 레이블을 생성
        real_label = torch.full((current_batch_size, 1), 1, dtype=torch.float32, device=device)
        fake_label = torch.full((current_batch_size, 1), 0, dtype=torch.float32, device=device)

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(current_batch_size, noise_size, 1, 1, device=device)
        fake_images = generator(z)
        g_loss = criterion(discriminator(fake_images), real_label)
        g_loss.backward()
        g_optimizer.step()

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = criterion(discriminator(images.to(device)), real_label)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # 손실 누적
        g_loss_accum += g_loss.item()
        d_loss_accum += d_loss.item()

        # 정확도 계산
        real_images = images.to(device)
        real_validity = discriminator(real_images)
        fake_validity = discriminator(fake_images.detach())
        correct_real += torch.sum(real_validity > 0.5).item()
        correct_fake += torch.sum(fake_validity < 0.5).item()
        total += current_batch_size

    # 에포크 종료 - 정확도 계산
    accuracy = 100 * (correct_real + correct_fake) / (2 * total)

    # 에포크당 평균 손실 계산
    avg_g_loss = g_loss_accum / len(data_loader)
    avg_d_loss = d_loss_accum / len(data_loader)

    # 시간 기록
    end_time = time.time()
    epoch_time = end_time - start_time

    # 출력 및 CSV 파일에 기록
    print(f"Epoch [{epoch+1}/{num_epoch}] - d_loss: {avg_d_loss:.5f}, g_loss: {avg_g_loss:.5f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_g_loss, avg_d_loss, epoch_time, accuracy])
