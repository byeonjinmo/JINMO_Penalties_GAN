import matplotlib.pyplot as plt
import pandas as pd

# 데이터 불러오기
data = pd.read_csv('/Users/mac/Desktop/11월16일까지 어케든 완성시켜/지표/FL0_2000(0)_resultss.csv')
new_data = pd.read_csv('/Users/mac/Desktop/11월16일까지 어케든 완성시켜/지표/FL0_2000_300_1600_(0-1-2)0.3_0.5_resultss.csv')

# 1000 에포크까지의 데이터만 선택
data = data.loc[data['Epoch'] <= 1000]
new_data = new_data.loc[new_data['Epoch'] <= 1000]

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 원래 데이터 (점선, 찐한 파랑색, 굵은 선)
plt.plot(data['Epoch'], data['Average D Accuracy'], label='Tradtional D Accuracy (Original)', color='darkblue', linestyle='dotted', linewidth=2)

# 새로운 데이터 (실선, 찐한 파랑색, 굵은 선)
plt.plot(new_data['Epoch'], new_data['Average D Accuracy'], label='Proposed D Accuracy (New)', color='darkblue', linestyle='solid', linewidth=2)

# 그래프 설정
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Comparison of Average D Accuracy over Epochs (Updated)')
plt.legend(fontsize=20)
plt.grid(True)
plt.show()
