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


temperature_df['날짜'] = temperature_df['날짜'].dt.to_period('W')

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


corr1 = combined_df['주간별진료'].corr(combined_df['주간일교차평균'])
corr2 = combined_df['주간별진료'].corr(combined_df['전주비교온도차'])

print(combined_df)

print(corr1)
print(corr2)
