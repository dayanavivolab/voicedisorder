import os, pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

'''
Description: Functions for drawing TSNE/UMAP projection of feature vectors 

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''


sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)

direc = 'data/path_phrase_hubert_fold1/embeddings'

x, y, x1, y1= [], [], [], []
for name in os.listdir(direc):
    with open(direc+'/'+name,'rb') as fid:
        feature, label = pickle.load(fid)
        x.append(np.mean(feature,axis=0))
        y.append(label)
        for i in range(0,len(feature)):
            x1.append(feature[i])
            y1.append(label)
                
X = np.array(x)
Y = np.array(y)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X)
print(X_embedded.shape)

X1 = np.array(x1)
Y1 = np.array(y1)
X1_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X1)
print(X1_embedded.shape)

plt.figure(1)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=Y, legend='full', palette=palette)
plt.figure(2)
sns.scatterplot(X1_embedded[:,0], X1_embedded[:,1], hue=Y1, legend='full', palette=palette)
plt.show()




