import matplotlib.pyplot as plt
import pandas as pd

# 데이터 불러오기 (실제 파일 위치에 맞게 경로 조정 필요)
gan_data = pd.read_csv('your_path_.csv').head(1000)

# 그래프 그리기
plt.figure(figsize=(15, 10))
# average Generator loss, average Discriminator loss (csv 열 이름 확인)
plt.plot(gan_data['Epoch'], gan_data['Average G Loss'], label='Average Generator Loss', color='blue'), #marker='s', markevery=20) # 마커,단위
plt.plot(gan_data['Epoch'], gan_data['Average D Loss'], label='Average Discriminator Loss', color='red'), #marker='s', markevery=20)

# 제목, 라벨, 범례 추가
#plt.title('Generator and Discriminator Loss Comparison (First 1000 Epochs)', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.ylabel('Average Loss', fontsize=25)
plt.legend(fontsize=20)
# x축과 y축 눈금 라벨의 크기 조정
plt.tick_params(axis='x', labelsize=20)  # x축
plt.tick_params(axis='y', labelsize=20)  # y축

plt.show()
