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
    plt.ion()
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
    plt.ion()
    plt.show()

def scattlerplot_3d(col1, col2, col3, data, number_of_clusters):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['purple', 'red', 'blue', 'green', 'yellow']
    for i in range(0, number_of_clusters):
        ax.scatter(data[col1][data.label == i], data[col2][data.label == i], data[col3][data.label == i], c=colors[i], s=60)
    ax.view_init(35, 185)
    plt.xlabel(col1)
    plt.ylabel(col2)
    ax.set_zlabel(col3)
    plt.ion()
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
    plt.ion()
    plt.show()

def label_clusters(number_of_clusters, data, limited_data):
    km = KMeans(number_of_clusters)
    km.fit(limited_data)
    labels = km.predict(limited_data)
    data["label"] = labels
    return data

def get_clustered_ids(data, number_of_clusters, id_row_name):
    for i in range(0, number_of_clusters):
        row = data[data["label"] == i]
        print(f'Number of rows in group_{i} = {len(row)}')
        print(row[id_row_name].values)
        print('=======================================')  


print("Welcome to market-segmantation!")
print("If you are unsure how to use the program, write 'help'!")
data = None
line = input()
while line != "quit":
    tokens = line.split(";")

    if tokens[0] == "help":
        print("This program gives you tools to analyze and clusterize data.")
        print("Using the comand line, you should input lines in the following format:")
        print("<command_name>;<arg1>;<arg2>;...<argN>")
        print("List of commands: ")
        print("  * read_data;<path_to_csv_data_file> // loads the data into working memory")
        print("  * get_distribution;<column_name> // gives you a distribution of the data by choosing a column")
        print("  * get_elbowcurve;<max_clusters>;<column1>;...<columnN> // displays elbow curve based on maximum number of clusters and N columns, used in calculating WCSS")
        print("  * get_clusters_2d;<number_of_clusters>;<column1>;<column2>;<id_column_name> // displays a 2d plot with colored clusters, and gives information about the row ids by cluster")
        print("  * get_clusters_3d;<number_of_clusters>;<column1>;<column2>;<column3>;<id_column_name> // displays a 3d plot with colored clusters, and gives information about the row ids by cluster")
        print("  * quit // exits the program")
        line = input()
        continue
    elif tokens[0] == "read_data":
        data = read_data(tokens[1])
        print("Successfully read data from csv!")
        line = input()
        continue

    if data is None:
        print("Please load data before doing further operations!")
        line = input()
        continue

    if tokens[0] == "get_distribution":
        get_distribution_by_column(tokens[1].strip(), data)
    elif tokens[0] == "get_elbowcurve":
        limited_data = limit_columns(tokens[2:], data)
        elbowcurve(int(tokens[1]), limited_data)
    elif tokens[0] == "get_clusters_2d":
        limited_data = limit_columns(tokens[2:], data)
        labeled_data = label_clusters(int(tokens[1]), data, limited_data)
        scatterplot_2d(tokens[2], tokens[3], labeled_data, True)
        get_clustered_ids(labeled_data, int(tokens[1]), tokens[4])
    elif tokens[0] == "get_clusters_3d":
        limited_data = limit_columns(tokens[2:], data)
        labeled_data = label_clusters(int(tokens[1]), data, limited_data)
        scattlerplot_3d(tokens[2], tokens[3], tokens[4], labeled_data, int(tokens[1]))
        get_clustered_ids(labeled_data, int(tokens[1]), tokens[5])
    else:
        print("Invalid command. Please check the doc, or use 'help'!")

    line = input()