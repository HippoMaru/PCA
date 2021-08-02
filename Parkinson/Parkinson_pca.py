import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def fill(value):
    if value == '-':
        return np.nan
    else:
        return float(value)


def clean(value):
    if 'Yes' in value:
        return 1
    else:
        return 0


df = pd.read_csv('dataset.csv')


features = df.drop(' Participant  code ', 1)
labels = df[' Participant  code ']

features['Gender'] = features['Gender'].map({'F': 0, 'M': 1})
features[' Positive  history  of  Parkinson  disease  in  family '] = features[' Positive  history  of  Parkinson  disease  in  family '].map({'No': 0, '-': 0.0375,'Yes': 1})
features[' Antidepressant  therapy '] = features[' Antidepressant  therapy '].apply(lambda x: clean(x))
features = features.drop(' Antidepressant  therapy ', 1)
features = features.drop(' Antiparkinsonian  medication ', 1)
features = features.drop(' Antipsychotic  medication ', 1)
features[' Benzodiazepine  medication '] = features[' Benzodiazepine  medication '].apply(lambda x: clean(x))
features = features.drop(' Levodopa  equivalent  (mg/day) ', 1)

feature_names = features.columns.tolist()
for column in feature_names:
    features[column] = features[column].apply(lambda x: fill(x))

imp = SimpleImputer(missing_values=np.nan, strategy='median', fill_value=None)
imp.fit(features)
features = pd.DataFrame(data=imp.transform(features), columns=features.columns)

for column in feature_names:
    print(column)
    print(features[column].value_counts(dropna=False))


features = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(features)
pca_features = ['principal component 1', 'principal component 2']
principalDf = pd.DataFrame(data=principalComponents, columns=pca_features)
finalDf = pd.concat([principalDf, labels], axis='columns')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
ClassLabels = ['PD', 'RBD', 'HC']
colors = ['r', 'g', 'b']
for label, color in zip(ClassLabels, colors):
    indicesToKeep = finalDf[' Participant  code '].apply(lambda x: label in x)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color, s=10)
ax.legend(ClassLabels)
ax.grid()
plt.show()
