# -*- coding: cp949 -*-


from imp import acquire_lock
from re import A
import pandas as pd

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


start_date = '2018-02-01'
end_date = '2019-07-31'

df_2018_filtered = air_df_2018[(air_df_2018['측정일시'] >= start_date) & (air_df_2018['측정일시'] <= end_date)]
df_2019_filtered = air_df_2019[(air_df_2019['측정일시'] >= start_date) & (air_df_2019['측정일시'] <= end_date)]

air_df_filtered = pd.concat([df_2018_filtered[['측정일시', 'average_pm10', 'average_pm25']], df_2019_filtered[['측정일시', 'average_pm10', 'average_pm25']]])

air_df_filtered = air_df_filtered.drop_duplicates()

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
print(air_df_grouped)


temperature_df['주간평균기온차'] = temperature_df.groupby('날짜')['평균기온차'].transform('mean')

temperature_df['주간일교차평균'] = temperature_df.groupby('날짜')['일교차'].transform('mean')

# New dataframe for calculated.
temperature_range_df = pd.DataFrame(columns = ['주간별', '주간일교차평균', '전주비교온도차'])


# fill columns

temperature_range_df['주간별'] = temperature_df['날짜']

temperature_range_df['주간일교차평균'] = temperature_df['주간일교차평균']

temperature_range_df['전주비교온도차'] = temperature_df['주간평균기온차'] 

temperature_range_df = temperature_range_df.drop_duplicates()





# Change column name.
treatment_df = treatment_df.rename(columns={'요양개시일': '주간별'})

# Concatenate two dataframes.
combined_df = pd.concat([treatment_df, temperature_range_df], axis = 1)

combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]


combined_df = combined_df.reset_index(drop=True)

print(combined_df)

combined_df = pd.concat([combined_df, air_df_grouped], axis = 1)



corr1 = combined_df['주간별진료'].corr(combined_df['주간일교차평균'])
corr2 = combined_df['주간별진료'].corr(combined_df['전주비교온도차'])
corr3 = combined_df['주간별진료'].corr(combined_df['주간미세먼지'])
corr4 = combined_df['주간별진료'].corr(combined_df['주간초미세먼지'])

print(corr1)
print(corr2)
print(corr3)
print(corr4)