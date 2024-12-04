# -*- coding: cp949 -*-


import pandas as pd

import chardet

with open('./dataFile.csv', 'rb') as file:
    result = chardet.detect(file.read())


df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding'])

df['요양개시일'] = pd.to_datetime(df['요양개시일'])

df['월별'] = df['요양개시일'].dt.to_period('M')

monthly_data = df.groupby('월별')['진료에피소드건수'].sum()



# 결과 출력
print(monthly_data)