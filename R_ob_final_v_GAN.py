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
num_epoch = 500
batch_size = 128
learning_rate = 0.0002
img_size = 32
num_channel = 3  # CIFAR-10 has 3 channels
dir_name = "+filter"
noise_size = 150

# 클래스 필터링 함수 정의
def filter_classes(dataset, exclude_classes):
    filtered_indices = [i for i in range(len(dataset)) if dataset.targets[i] not in exclude_classes]
    return torch.utils.data.Subset(dataset, filtered_indices)

# 허 초기화 함수 정의
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
# CSV 파일 준비
csv_file = os.path.join(dir_name, "+filter.csv")
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Generator Loss", "Discriminator Loss", "Discriminator Accuracy on Real", "Discriminator Accuracy on Fake", "Time Taken (seconds)"])
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


# CIFAR-10 데이터셋 설정 및 필터링
CIFAR10_dataset = torchvision.datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=transform,
    download=True
)

# 배제할 클래스 목록
exclude_classes = [8, 9]

# 필터링된 데이터셋 생성
filtered_dataset = filter_classes(CIFAR10_dataset, exclude_classes)

# 필터링된 데이터셋으로 데이터 로더 생성
data_loader = torch.utils.data.DataLoader(
    dataset=filtered_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Declares discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(num_channel, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 10),
                                       nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


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

    def forward(self, x):
        x = x.view(-1, noise_size)  # 변경된 부분
        out = self.l1(x)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Initialize generator/Discriminator
discriminator = Discriminator().to(device)
generator = Generator().to(device)

# 허 초기화 적용
discriminator.apply(init_weights)
generator.apply(init_weights)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# 패널티 가중치 초기화
g_penalty_weight = 0.0
d_penalty_weight = 0.0

# Training part
initial_no_penalty_epochs = 0  # 처음 에포크 동안은 패널티 없음

# 페널티 카운터 초기화
g_penalty_count = 0
d_penalty_count = 0

# 연속적인 패널티가 특정 횟수 이상인 경우 추가 페널티 부여
if g_penalty_count > 4:
    g_penalty_weight += 0.5  # 생성자의 패널티를 추가로 증가
    g_penalty_count = 0  # 생성자의 연속 패널티 카운터 초기화
    print("생성자에 추가 페널티 부여")
if d_penalty_count > 4:
    d_penalty_weight += 0.5  # 판별자의 패널티를 추가로 증가
    d_penalty_count = 0  # 판별자의 연속 패널티 카운터 초기화
    print("판별자에 추가 페널티 부여")

# Training part
for epoch in range(num_epoch):
    #로계위
    g_loss_accum = 0.0
    d_loss_accum = 0.0
    start_time = time.time()
    #정계위
    correct_real = 0
    correct_fake = 0
    # 페널티 카운터의 카운터 초기화
    # p_count = 0

    # if p_count == 1:
    #     g_penalty_count = 0
    #     d_penalty_count = 0
    #     p_count = 0
    for i, (images, _) in enumerate(data_loader):
        current_batch_size = images.size(0)  # 현재 배치의 크기

        # 실제 및 가짜 레이블 생성
        real_label = torch.full((current_batch_size, 1), 1, dtype=torch.float32, device=device)
        fake_label = torch.full((current_batch_size, 1), 0, dtype=torch.float32, device=device)

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(current_batch_size, noise_size, 1, 1, device=device)
        fake_images = generator(z)
        g_loss = criterion(discriminator(fake_images), real_label) * g_penalty_weight #페널티 적용
        g_loss.backward()
        g_optimizer.step()

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = criterion(discriminator(images.to(device)), real_label)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2 * d_penalty_weight #페널티 적용
        d_loss.backward()
        d_optimizer.step()

        # 실제 이미지에 대한 판별자의 정확도 계산 (실제 이미지를 "진짜"로 정확하게 구분 척도 )
        real_output = discriminator(images.to(device))
        correct_real += (real_output > 0.5).sum().item()

        # 가짜 이미지에 대한 판별자의 정확도 계산 (생성된 이미지를 "가짜"로 정확하게 구분 척도)
        fake_output = discriminator(fake_images.detach())
        correct_fake += (fake_output < 0.5).sum().item()

        # 손실 누적
        g_loss_accum += g_loss.item()
        d_loss_accum += d_loss.item()
        #만약 데이터 필터시 수치 잘 보이도록 배치 사이즈 수정  ex)2로 나눠서 전체 크기 확인 후 변경 추천
        if (i + 1) % 313 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                  .format(epoch, num_epoch, i + 1, len(data_loader), d_loss.item(), g_loss.item()))

        if (i + 1) % 313 == 0:
            with torch.no_grad():
                z = torch.randn(100, noise_size, 1, 1, device=device)  # 100개의 노이즈 벡터 생성
                fake_images = generator(z)
                samples = fake_images.reshape(100, num_channel, img_size, img_size)
                save_image(samples, os.path.join(dir_name, f'GAN_fake_samples_epoch{epoch + 1}_batch{i + 1}.png'),
                           nrow=10)

    # 에포크당 평균 손실 계산
    avg_g_loss = g_loss_accum / len(data_loader)
    avg_d_loss = d_loss_accum / len(data_loader)

    # 패널티 조정 로직
    if epoch > initial_no_penalty_epochs and epoch % 40 == 0:
        if avg_g_loss < avg_d_loss:
            g_penalty_weight += 0.5
            g_penalty_count += 1
            d_penalty_weight = max(1.0, d_penalty_weight - 0.001)
            print(f"Epoch {epoch}: 후훗 이런 이런..나는 앞서가는 자를 좋아하지 않아. 생성자 네놈에게 페널티를 부여하지..")
        else:
            d_penalty_weight += 0.5
            d_penalty_count += 1
            g_penalty_weight = max(1.0, g_penalty_weight - 0.001)
            print(f"Epoch {epoch}: 후훗 이런 이런..나는 앞서가는 자를 좋아하지 않아. 판별자 네놈에게 페널티를 부여하지..")


    # 에포크당 판별자의 정확도 계산 및 출력
    accuracy_real = correct_real / (len(data_loader) * batch_size)
    accuracy_fake = correct_fake / (len(data_loader) * batch_size)
    print(
    f"Epoch {epoch}: Discriminator Accuracy on Real Images: {accuracy_real:.2f}, on Fake Images: {accuracy_fake:.2f}")

    # 시간 기록
    end_time = time.time()
    epoch_time = end_time - start_time

    # CSV 파일에 기록
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_g_loss, avg_d_loss, accuracy_real, accuracy_fake, epoch_time])