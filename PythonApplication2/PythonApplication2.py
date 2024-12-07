# -*- coding: cp949 -*-


import pandas as pd

import chardet

# dataFile.csv: Number of atopic dermatitis treatments
with open('./dataFile.csv', 'rb') as file:
    result = chardet.detect(file.read())

# dataFile2.csv: Average, Maximum, Minimum temperature of Korea. Average and daily temperature range (Max-Min) will be used.
with open('./dataFile2.csv', 'rb') as file2:
    result2 = chardet.detect(file2.read())

treatment_df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding']) 

temperature_df = pd.read_csv('./dataFile2.csv', header = 6, encoding = result['encoding'])

# change date to datetime.
treatment_df['��簳����'] = pd.to_datetime(treatment_df['��簳����'])


temperature_df['��¥'] = pd.to_datetime(temperature_df['��¥'])



# add a new column to calculate the difference in average temperature and daily temperature change.
temperature_df['������ձ��'] =  temperature_df['��ձ��(��)'].shift(1)

temperature_df['��ձ����'] = (temperature_df['��ձ��(��)'] - temperature_df['������ձ��']).abs()

temperature_df['�ϱ���'] = temperature_df['�ְ���(��)'] - temperature_df['�������(��)']

# Put the data together on a weekly basis.
treatment_df['��簳����'] = treatment_df['��簳����'].dt.to_period('W')

treatment_df['�ְ�������'] = treatment_df.groupby('��簳����')['���ῡ�Ǽҵ�Ǽ�'].transform('sum')

treatment_df = treatment_df.drop(columns=['���ῡ�Ǽҵ�Ǽ�'])

treatment_df = treatment_df.drop_duplicates()


temperature_df['��¥'] = temperature_df['��¥'].dt.to_period('W')

temperature_df['�ְ���ձ����'] = temperature_df.groupby('��¥')['��ձ����'].transform('mean')

temperature_df['�ְ��ϱ������'] = temperature_df.groupby('��¥')['�ϱ���'].transform('mean')

# New dataframe for calculated.
temperature_range_df = pd.DataFrame(columns = ['�ְ���', '�ְ��ϱ������', '���ֺ񱳿µ���'])


# fill columns

temperature_range_df['�ְ���'] = temperature_df['��¥']

temperature_range_df['�ְ��ϱ������'] = temperature_df['�ְ��ϱ������']

temperature_range_df['���ֺ񱳿µ���'] = temperature_df['�ְ���ձ����'] 

temperature_range_df = temperature_range_df.drop_duplicates()






# Change column name.
treatment_df = treatment_df.rename(columns={'��簳����': '�ְ���'})


# Concatenate two dataframes.
combined_df = pd.concat([treatment_df, temperature_range_df], axis = 1)

combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]


corr1 = combined_df['�ְ�������'].corr(combined_df['�ְ��ϱ������'])
corr2 = combined_df['�ְ�������'].corr(combined_df['���ֺ񱳿µ���'])

print(combined_df)

print(corr1)
print(corr2)
