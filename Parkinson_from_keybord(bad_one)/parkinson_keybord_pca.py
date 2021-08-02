import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pd.set_option('display.max_columns', None)
users_files = glob.glob(r'./dataset/Archived-users/Archived users/*.txt', recursive=True)

users_data = []
i = 0

for file in users_files:
    with open(file) as f:
        users_data.append({})
        for line in f.readlines():
            key, value = line.split(': ')
            users_data[i][key] = value.strip()
        users_data[i]['ID'] = os.path.basename(file)[-14:-4]
        i += 1

users_df = pd.DataFrame(users_data)
users_df = users_df.set_index('ID')
users_df.replace('------', np.nan, inplace=True)
users_df.replace('', np.nan, inplace=True)

boolean_features = ['Levadopa', 'MAOB', 'DA', 'Parkinsons', 'Tremors', 'Tremors', 'Other']
for boolean_feature in boolean_features:
    users_df[boolean_feature] = users_df[boolean_feature].map({'False': 0, 'True': 1})
users_df['Gender'] = users_df['Gender'].map({'Female': 0, 'Male': 1})


keyboards_files = glob.glob(r'./dataset/Archived-Data/Tappy Data/*.txt', recursive=True)

keyboards_data = []
keys = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']
i = 0

for file in keyboards_files:
    with open(file) as f:
        keyboards_data.append({})
        for line in f.readlines():
            values = line.split('\t')
            for j in range(len(keys)):
                try:
                    keyboards_data[i][keys[j]] = values[j]
                except IndexError:
                    i -= 1
                    break
        i += 1

keyboards_df = pd.DataFrame(keyboards_data)
keyboards_df = keyboards_df.set_index('ID')

users_df = users_df[users_df['BirthYear'].notna()]
users_df = users_df[users_df['DiagnosisYear'].notna()]
users_df = users_df[users_df['Impact'].notna()]
users_df = users_df.drop('Tremors', 1)
users_df = users_df.drop('UPDRS', 1)
users_df['Sided'] = users_df['Sided'].map({'Left': -1, 'None': 0, 'Right': 1})
users_df['Impact'] = users_df['Impact'].map({'Mild': -1, 'Medium': 0, 'Severe': 1})

valid_members = users_df.index.tolist()
print(users_df)
feature_names = users_df.columns.tolist()
for column in feature_names:
    print(column)
    print(users_df[column].value_counts(dropna=False))


'''использовать данные с клавиатуры я не могу, ибо не понимаю, от чего именно искать зависимость'''
'''данные об участниках к дальнейшей обработке готовы, к ним нужно добавить признаки, взятые с данных'''
'''нажатий клавиатуры, но какие именно я не понимаю'''
