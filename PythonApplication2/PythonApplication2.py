# -*- coding: cp949 -*-


from imp import acquire_lock
from platform import machine
from re import A
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics


import chardet

import matplotlib.pyplot as plt

# dataFile.csv: Number of atopic dermatitis treatments
with open('./dataFile.csv', 'rb') as file:
    result = chardet.detect(file.read())
    
# dataFile2.csv: Average, Maximum, Minimum temperature of Korea. Average and daily temperature range (Max-Min) will be used.
with open('./dataFile2.csv', 'rb') as file2:
    result2 = chardet.detect(file2.read())

# datafile 3,4: Air pollution data(Seoul)
with open('./dataFile3.csv', 'rb') as file3:
    result3 = chardet.detect(file3.read())
with open('./dataFile4.csv', 'rb') as file4:
    result4 = chardet.detect(file4.read())


treatment_df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding']) 

temperature_df = pd.read_csv('./dataFile2.csv', header = 6, encoding = result2['encoding'])

air_df_2018 = pd.read_csv('./dataFile3.csv', header = 0, encoding = result3['encoding'])
air_df_2019 = pd.read_csv('./dataFile4.csv', header = 0, encoding = result4['encoding'])

air_df_2018['측정일시'] = pd.to_datetime(air_df_2018['측정일시'], format='%Y%m%d')
air_df_2019['측정일시'] = pd.to_datetime(air_df_2019['측정일시'], format='%Y%m%d')

pm_columns_2018 = [col for col in air_df_2018.columns if '미세먼지' in col]  
pm_columns_2019 = [col for col in air_df_2019.columns if '미세먼지' in col]  

air_df_2018['average_pm10'] = air_df_2018[pm_columns_2018].mean(axis=1, skipna=True)
air_df_2019['average_pm10'] = air_df_2019[pm_columns_2019].mean(axis=1, skipna=True)

pm_columns_2018_2 = [col for col in air_df_2018.columns if '초미세' in col]  
pm_columns_2019_2 = [col for col in air_df_2019.columns if '초미세' in col]  

air_df_2018['average_pm25'] = air_df_2018[pm_columns_2018_2].mean(axis=1, skipna=True)
air_df_2019['average_pm25'] = air_df_2019[pm_columns_2019_2].mean(axis=1, skipna=True)



df_2018_filtered = air_df_2018[(air_df_2018['측정일시'] >= '2018-02-01') & (air_df_2018['측정일시'] <= '2018-12-31')]
df_2019_filtered = air_df_2019[(air_df_2019['측정일시'] >= '2019-01-01') & (air_df_2019['측정일시'] <= '2019-07-31')]



air_df_filtered = pd.concat([df_2018_filtered[['측정일시', 'average_pm10', 'average_pm25']], df_2019_filtered[['측정일시', 'average_pm10', 'average_pm25']]])


air_df_filtered = air_df_filtered.drop_duplicates()
air_df_filtered = air_df_filtered.dropna()

air_df_grouped = air_df_filtered.groupby('측정일시').mean()

# change date to datetime.
treatment_df['요양개시일'] = pd.to_datetime(treatment_df['요양개시일'])


temperature_df['날짜'] = pd.to_datetime(temperature_df['날짜'])


# add a new column to calculate the difference in average temperature and daily temperature change.
temperature_df['전일평균기온'] =  temperature_df['평균기온(℃)'].shift(1)

temperature_df['평균기온차'] = (temperature_df['평균기온(℃)'] - temperature_df['전일평균기온']).abs()

temperature_df['일교차'] = temperature_df['최고기온(℃)'] - temperature_df['최저기온(℃)']

# Put the data together on a weekly basis.
treatment_df['요양개시일'] = treatment_df['요양개시일'].dt.to_period('W')

treatment_df['주간별진료'] = treatment_df.groupby('요양개시일')['진료에피소드건수'].transform('sum')

treatment_df = treatment_df.drop(columns=['진료에피소드건수'])

treatment_df = treatment_df.drop_duplicates()


air_df_grouped = air_df_grouped.reset_index()
air_df_grouped['측정일시'] = pd.to_datetime(air_df_grouped['측정일시'])
air_df_grouped['측정일시'] = air_df_grouped['측정일시'].dt.to_period('W')

air_df_grouped['주간미세먼지'] = air_df_grouped.groupby('측정일시')['average_pm10'].transform('mean')

air_df_grouped['주간초미세먼지'] = air_df_grouped.groupby('측정일시')['average_pm25'].transform('mean')
air_df_grouped = air_df_grouped.drop('average_pm10', axis = 1)
air_df_grouped = air_df_grouped.drop('average_pm25', axis = 1)

temperature_df['날짜'] = temperature_df['날짜'].dt.to_period('W')
air_df_grouped = air_df_grouped.drop_duplicates()

air_df_grouped = air_df_grouped.reset_index(drop=True)

temperature_df['주간평균기온차'] = temperature_df.groupby('날짜')['평균기온차'].transform('mean')

temperature_df['주간일교차평균'] = temperature_df.groupby('날짜')['일교차'].transform('mean')

# New dataframe for calculated.
temperature_range_df = pd.DataFrame(columns = ['주간별', '주간일교차평균', '전주비교온도차'])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# fill columns

temperature_range_df['주간별'] = temperature_df['날짜']

temperature_range_df['주간일교차평균'] = temperature_df['주간일교차평균']

temperature_range_df['전주비교온도차'] = temperature_df['주간평균기온차'] 

temperature_range_df = temperature_range_df.drop_duplicates()




# Change column name.
treatment_df = treatment_df.rename(columns={'요양개시일': '주간별'})

# Concatenate two dataframes.
columns_to_use = ['주간일교차평균', '전주비교온도차']
combined_df = pd.concat([treatment_df, temperature_range_df[columns_to_use]], axis= 1)
combined_df = combined_df.reset_index(drop=True)


columns_to_use = ['주간미세먼지', '주간초미세먼지']
combined_df = pd.merge(combined_df, air_df_grouped, left_on='주간별', right_on='측정일시', how='inner')

combined_df = combined_df.drop('측정일시', axis = 1)

combined_df = combined_df.sort_values(by='주간별').reset_index(drop=True)


plt.figure(figsize=(10, 6))
plt.bar(combined_df.index, combined_df['주간별진료'], color='blue', alpha=0.7)
plt.title('Number of treatment')
plt.xlabel('Index (Row)')
plt.ylabel('Number of treatment')
plt.xticks(rotation=90)  # x축 레이블이 겹치지 않도록 회전
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(combined_df.index, combined_df['주간일교차평균'], color='blue', alpha=0.7)
plt.title('Temperature difference of day')
plt.xlabel('Index (Row)')
plt.ylabel('Temperature difference of day')
plt.xticks(rotation=90)  # x축 레이블이 겹치지 않도록 회전
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(combined_df.index, combined_df['전주비교온도차'], color='blue', alpha=0.7)
plt.title('Temperature difference of previous day')
plt.xlabel('Index (Row)')
plt.ylabel('Temperature difference of previous day')
plt.xticks(rotation=90)  # x축 레이블이 겹치지 않도록 회전
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(combined_df.index, combined_df['주간미세먼지'], color='blue', alpha=0.7)
plt.title('weekly fine dust')
plt.xlabel('Index (Row)')
plt.ylabel('weekly fine dust')
plt.xticks(rotation=90)  # x축 레이블이 겹치지 않도록 회전
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
plt.bar(combined_df.index, combined_df['주간초미세먼지'], color='blue', alpha=0.7)
plt.title('weekly ultrafine dust')
plt.xlabel('Index (Row)')
plt.ylabel('weekly ultrafine dust')
plt.xticks(rotation=90)  # x축 레이블이 겹치지 않도록 회전
plt.grid(True)
plt.show()

corr1 = combined_df['주간별진료'].corr(combined_df['주간일교차평균'])
corr2 = combined_df['주간별진료'].corr(combined_df['전주비교온도차'])
corr3 = combined_df['주간별진료'].corr(combined_df['주간미세먼지'])
corr4 = combined_df['주간별진료'].corr(combined_df['주간초미세먼지'])

print("corr of temperature diff of day:", end='')
print(corr1)
print("corr of temperature diff of previous day:", end='')
print(corr2)
print("corr of temperature diff of fine dust:", end='')
print(corr3)
print("corr of temperature diff of ultrafine dust:", end='')
print(corr4)



# this is for machine learning to expect when will be crowded for treatment.
machineLearning_df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding']) 

machineLearning_df['요양개시일'] = pd.to_datetime(machineLearning_df['요양개시일'])

# this deletes outlier, which will include Sunday and holiday.
machineLearning_df = machineLearning_df[~(machineLearning_df['진료에피소드건수'] < 1000) ]

# this defines 'crowded' for above mean.
mean_visits = machineLearning_df['진료에피소드건수'].mean()

machineLearning_df['몰림'] = (machineLearning_df['진료에피소드건수'] >= mean_visits).astype(int)

# feature
X = machineLearning_df['요양개시일']

# target
y = machineLearning_df['몰림']

# divide train data and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values.reshape(-1, 1)

X_test = X_test.values.reshape(-1, 1)

# data normalization.
scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_scaled, y_train)

y_hat = knn.predict(X_test_scaled)

# 정확도 및 성능 평가
print("Accuracy:", accuracy_score(y_test, y_hat))
print(classification_report(y_test, y_hat))


knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)



# 혼동 행렬을 시각화
plt.figure(figsize=(6, 6))
sns.heatmap(knn_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 2. Precision, Recall, F1-Score 시각화
report = classification_report(y_test, y_hat, output_dict=True)

# Precision, Recall, F1-score을 각 클래스별로 저장
labels = ['0', '1']
precision = [report['0']['precision'], report['1']['precision']]
recall = [report['0']['recall'], report['1']['recall']]
f1_score = [report['0']['f1-score'], report['1']['f1-score']]

# 각 성능 지표 그래프
x = np.arange(len(labels))  # 클래스 0, 1

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.2  # 바의 너비
ax.bar(x - width, precision, width, label='Precision', color='lightblue')
ax.bar(x, recall, width, label='Recall', color='lightgreen')
ax.bar(x + width, f1_score, width, label='F1-Score', color='lightcoral')

ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Precision, Recall, F1-Score by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
