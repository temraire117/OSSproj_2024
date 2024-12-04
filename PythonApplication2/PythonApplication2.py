# -*- coding: cp949 -*-


import pandas as pd

import chardet

with open('./dataFile.csv', 'rb') as file:
    result = chardet.detect(file.read())


df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding'])

df['��簳����'] = pd.to_datetime(df['��簳����'])

df['����'] = df['��簳����'].dt.to_period('M')

monthly_data = df.groupby('����')['���ῡ�Ǽҵ�Ǽ�'].sum()



# ��� ���
print(monthly_data)