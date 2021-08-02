import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


dataset = pd.read_csv('dataset_no_outlier.csv')

data = dataset.loc[:, ['X', 'Y', 'Z', 'Mixed']].values
target = dataset.loc[:, 'ClassLabel'].values

data = StandardScaler().fit_transform(data)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
features = ['principal component 1', 'principal component 2']
principalDf = pd.DataFrame(data=principalComponents, columns=features)
finalDf = pd.concat([principalDf, dataset[['ClassLabel']]], axis='columns')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
ClassLabels = [1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'k', 'm']
for target, color in zip(ClassLabels, colors):
    indicesToKeep = finalDf['ClassLabel'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color, s=10)
ax.legend(ClassLabels)
ax.grid()
plt.show()
