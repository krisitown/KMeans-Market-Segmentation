#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
#%matplotlib inline

def read_data(filePath): 
    return pd.read_csv(filePath)

def get_distribution_by_column(col, data):
    plt.figure(figsize=(10, 6))
    sns.set(style = 'whitegrid')
    sns.distplot(data[col])
    plt.title('Distribution of ' + col, fontsize = 20)
    plt.xlabel('Range of ' + col)
    plt.ylabel('Count')
    plt.show()

def limit_columns(columns, data):
    tmpDataFrame = data[data.columns]
    return tmpDataFrame[columns]

def scatterplot_2d(col1, col2, data, labeled):
    plt.figure(figsize=(10,6))
    if(labeled):
        sns.scatterplot(x = col1,y = col2, hue="label", data = data,  
        legend='full',s = 60)
    else:
        sns.scatterplot(x = col1,y = col2,  data = data[[col1, col2]]  ,s = 60 )
    plt.xlabel(col1)
    plt.ylabel(col2) 
    plt.title(col2 + ' vs ' + col1)
    plt.show()

def wcss(max_clusters, data):
    wcss=[]
    for i in range(1,max_clusters):
        km=KMeans(n_clusters=i)
        km.fit(data)
        wcss.append(km.inertia_)
    return wcss

def elbowcurve(max_clusters, data):
    within_cluster_sum_squares = wcss(max_clusters, data)
    plt.figure(figsize=(12,6))
    plt.plot(range(1,max_clusters),within_cluster_sum_squares)
    plt.plot(range(1,max_clusters),within_cluster_sum_squares, linewidth=2, color="red", marker ="8")
    plt.xlabel("K Value")
    plt.xticks(np.arange(1,max_clusters,1))
    plt.ylabel("WCSS")
    plt.show()

def label_clusters(number_of_clusters, data, limited_data):
    km = KMeans(number_of_clusters)
    km.fit(limited_data)
    labels = km.predict(limited_data)
    data["label"] = labels
    return data

data = read_data("market_data.csv")
X = limit_columns(["Annual Income (k$)", "Spending Score (1-100)"], data)
#elbowcurve(11, X)
#scatterplot_2d("Annual Income (k$)", "Spending Score (1-100)", data)
#get_distribution_by_column('Age', data)
labeled_data = label_clusters(5, data, X)
print(labeled_data.head())
scatterplot_2d("Annual Income (k$)", "Spending Score (1-100)", labeled_data, True)