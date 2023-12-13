import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import csv


# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "GAN_results_PSTx_+pn(0.5)+drop(0.25)"

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024
# 점진적 학습을 위한 히든 사이즈 새로 구성
hidden_size0 = 64
hidden_size01 = 128

# 패널티 가중치 초기화
g_penalty_weight = 1.0
d_penalty_weight = 1.0

# 점진적 단계 학습을 위한 설정

epoch = 0
class opt:
    step = 0
    if epoch > 50:
        step = 1
    elif epoch > 100:
        step = 2
    elif epoch > 150:
        step = 3

# 모델 변경시 가중치 초기화 메커니즘 / 선형혼합 방식 약간 변경
def mix_weights(old_model, new_model, alpha=0.1):
    old_weights = old_model.state_dict()
    new_weights = new_model.state_dict()

    mixed_weights = {
        name: (1 - alpha) * old_weights[name] + alpha * new_weights[name]
        for name in old_weights
    }

    new_model.load_state_dict(mixed_weights)
    return new_model

 # 렐루 , 리키렐루를 사용함으로 허 초기화 적
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))


# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

# MNIST dataset setting
MNIST_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# Declares discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(img_size, hidden_size3)
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, 1)
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(0.25)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
# 점진적 단계 학습의 1단계 / 쉬운 모델 레이어 중 제일 낮은 레이어 크기를 128로 변경
class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()

        self.linear1 = nn.Linear(img_size, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, hidden_size01)
        self.linear4 = nn.Linear(hidden_size01, 1)
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(0.25)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
# 점진적 단계 학습의 2단계 / 쉬운 모델 레이어 중 제일 낮은 레이어 크기를 64로 변경
class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.linear1 = nn.Linear(img_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size01)
        self.linear3 = nn.Linear(hidden_size01, hidden_size0)
        self.linear4 = nn.Linear(hidden_size0, 1)
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(0.25)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

# Declares generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, img_size)
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x
# 점진적 단계 학습의 1단계 / 쉬운 모델 레이어 중 제일 낮은 레이어 크기를 128로 변경
class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size01)
        self.linear2 = nn.Linear(hidden_size01, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, img_size)
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x

# 점진적 단계 학습의 2단계 / 쉬운 모델 레이어 중 제일 낮은 레이어 크기를 64로 변경
class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size0)
        self.linear2 = nn.Linear(hidden_size0, hidden_size01)
        self.linear3 = nn.Linear(hidden_size01, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, img_size)
        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x

# Initialize generator/Discriminator
discriminator = Discriminator2()
generator = Generator1()
# Device setting
discriminator = discriminator.to(device)
generator = generator.to(device)
# 각각 허 초기화 모델들에 적용 .apply(weights_init)  적용 실패 ^_< 자꾸 오류남 ㅋㅋㅋ ㅜ
# generator.apply(weights_init)
# discriminator.apply(weights_init)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Training part
initial_no_penalty_epochs = 0  # 처음 에포크 동안은 패널티 없음
"""
Training part
"""
# 정확한 모델 판별을 위한 각 손실 밑 퍼포먼스 값 csv파일로 저장하기 위한 코드
with open('gan_training_metrics_PSTx_+pn(0.5)+drop(0.25).csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Epoch', 'D_Loss', 'G_Loss', 'D_Performance', 'G_Performance'])
    for epoch in range(num_epoch):
        # 점진적 모델 변경 실제 학습 루프에 적용
        if opt.step == 1:
            # Transition for Generator
            new_generator = Generator1()
            generator = new_generator
            generator = mix_weights(generator, new_generator, alpha=0.5)

            # Transition for Discriminator_
            # new_discriminator = Discriminator2()
            # discriminator = new_discriminator
            # discriminator = mix_weights(discriminator, new_discriminator, alpha=0.5)

        if opt.step == 2:
            new_generator = Generator()
            generator = mix_weights(generator, new_generator, alpha=0.5)

            new_discriminator = Discriminator1()
            discriminator = new_discriminator
            discriminator = mix_weights(discriminator, new_discriminator, alpha=0.5)

            # # Transition for Discriminator
            # new_discriminator = Discriminator()
            # discriminator = new_discriminator
            # discriminator = mix_weights(discriminator, new_discriminator, alpha=0.5)

        for i, (images, label) in enumerate(data_loader):

            # make ground truth (labels) -> 1 for real, 0 for fake
            real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

            # reshape real images from MNIST dataset
            real_images = images.reshape(batch_size, -1).to(device)

            # +---------------------+
            # |   train Generator   |
            # +---------------------+

            # Initialize grad
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # make fake images with generator & noise vector 'z'
            z = torch.randn(batch_size, noise_size).to(device)
            fake_images = generator(z)

            # Compare result of discriminator with fake images & real labels
            # If generator deceives discriminator, g_loss will decrease
            g_loss = criterion(discriminator(fake_images), real_label) * g_penalty_weight # 가중치 페널티를 적용을 위함

            # Train generator with backpropagation
            g_loss.backward()
            g_optimizer.step()

            # +---------------------+
            # | train Discriminator |
            # +---------------------+

            # Initialize grad
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            # make fake images with generator & noise vector 'z'
            z = torch.randn(batch_size, noise_size).to(device)
            fake_images = generator(z)

            # Calculate fake & real loss with generated images above & real images
            fake_loss = criterion(discriminator(fake_images), fake_label)
            real_loss = criterion(discriminator(real_images), real_label)
            d_loss = (fake_loss + real_loss) / 2 * d_penalty_weight

            # Train discriminator with backpropagation
            # In this part, we don't train generator
            d_loss.backward()
            d_optimizer.step()

            d_performance = discriminator(real_images).mean()
            g_performance = discriminator(fake_images).mean()
        # 패널티 조정 로직
        if epoch > initial_no_penalty_epochs and epoch % 2 == 0:
            if g_loss < d_loss:
                g_penalty_weight += 0.0002
                d_penalty_weight = max(1.0, d_penalty_weight - 0.0001)
                print(f"Epoch {epoch}: 생페.")
            else:
                d_penalty_weight += 0.0002
                g_penalty_weight = max(1.0, g_penalty_weight - 0.0001)
                print(f"Epoch {epoch}: 판페.")

            if (i + 1) % 150 == 0:
                print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                      .format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))

        # print discriminator & generator's performance
        print(" Epock {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"
              .format(epoch, d_performance, g_performance))

        # CSV파일에 값을 쓰는 코드
        writer.writerow([epoch, d_loss.item(), g_loss.item(), d_performance.item(), g_performance.item()])

        # Save fake images in each epoch
        samples = fake_images.reshape(batch_size, 1, 28, 28)
        save_image(samples, os.path.join(dir_name, 'GAN_fake_samples{}.png'.format(epoch + 1)))