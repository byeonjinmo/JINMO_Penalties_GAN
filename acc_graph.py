import matplotlib.pyplot as plt
import pandas as pd

# 데이터 불러오기
data = pd.read_csv('/Users/mac/Desktop/11월16일까지 어케든 완성시켜/지표/FL0_2000(0)_resultss.csv')
new_data = pd.read_csv('/Users/mac/Desktop/11월16일까지 어케든 완성시켜/지표/FL0_2000_300_1600_(0-1-2)0.3_0.5_resultss.csv')

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 원래 데이터 (점선, 찐한 파랑색, 굵은 선)
plt.plot(data['Epoch'], data['Average D Accuracy'], label='Average D Accuracy (Original)', color='darkblue', linestyle='dotted', linewidth=2)

# 새로운 데이터 (실선, 찐한 파랑색, 굵은 선)
plt.plot(new_data['Epoch'], new_data['Average D Accuracy'], label='Average D Accuracy (New)', color='darkblue', linestyle='solid', linewidth=2)

# 그래프 설정
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Comparison of Average D Accuracy over Epochs (Updated)')
plt.legend()
plt.grid(True)
plt.show()
