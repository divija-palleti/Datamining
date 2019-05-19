import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
# from sklearn import preprocessing
import pandas as pd


class K_Means:
    def __init__(self, k=5, tol=0.001, max_iter=3000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in X:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True
            for c in self.centroids:
                # print("lol")
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                print('Final Centeroids:')
                print(self.centroids)
                break



df = pd.read_csv('Mall_Customers.csv', index_col=0, encoding="utf-8-sig")

df.fillna(0,inplace=True)

df.loc[df['Gender'] == 'Male', 'Gender'] = 1
df.loc[df['Gender'] == 'Female', 'Gender'] = 0

print(df.head())


X = np.array(df.astype(float))
print(X.shape)
# print(X.shape)
clf = K_Means()
clf.fit(X)