import pandas as pd
import matplotlib.pyplot as plt

# csv 입력
file_path = 'your_file_path.csv'
data = pd.read_csv(file_path)

# 손실 그래프
plt.figure(figsize=(12, 6))
plt.plot(data['Epoch'], data['D_Loss'], label='Discriminator Loss', color='blue')
plt.plot(data['Epoch'], data['G_Loss'], label='Generator Loss', color='red')
plt.title('Discriminator vs Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 성능 그래프
plt.figure(figsize=(12, 6))
plt.plot(data['Epoch'], data['D_Performance'], label='Discriminator Performance', color='blue')
plt.plot(data['Epoch'], data['G_Performance'], label='Generator Performance', color='red')
plt.title('Discriminator vs Generator Performance')
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.legend()
plt.grid(True)
plt.show()
