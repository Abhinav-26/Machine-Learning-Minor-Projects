#importing all the libraries that will be required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#reading and storing all data from csv file to data
data=pd.read_csv("Appstore_Games.csv")

#dropping all the columns that are not required
data = data.drop("URL", axis=1)
data = data.drop("Name", axis=1)
data = data.drop("Subtitle", axis=1)
data = data.drop("Icon URL", axis=1)
data = data.drop("ID", axis=1)
data=data.drop("Developer", axis=1)
data=data.drop("Original Release Date", axis=1)
data=data.drop("Current Version Release Date", axis=1)
data=data.drop("Description",axis=1)
data=data.drop("Age Rating", axis=1)

#Checking the total values availabe in the columns :- average user rating and primary genre
len([i for i in list(data["Average User Rating"].isna()) if i==False])
len([i for i in data["Primary Genre"] if i=="Games"])


#Droping the rows with missing User Rating
data=data.dropna(subset=["Average User Rating"])
print(data.shape)


#Checking length of average user rating having rating greater than 4.0
len([ i for i in data["Average User Rating"] if i>=4.0])

#Checking length of average user rating having rating greater than 4.0 and having games genre
len([i for i in zip(data["Average User Rating"], data["Primary Genre"]) if i[0]>=4.0 and i[1]=="Games"])
print("Printing the genre and its count \n",data["Primary Genre"].value_counts())

print("\n",data.describe(include="all"))
genre_counts = dict(data['Primary Genre'].value_counts())

#distributed_genres dictionary is created
distributed_genres={}
for i in genre_counts.keys():
    distributed_genres[i]=data[data['Primary Genre']==i]

#printing the keys of the dictionary created
print("\n",distributed_genres.keys())


#Plotting of graph b/w games and average user rating
sns.distplot(distributed_genres['Games']['Average User Rating'], kde=False)


#Plotting histograph for all keys v/s Average User Rating present in our dictionary
for i in distributed_genres.keys():
    sns.distplot(distributed_genres[i]['Average User Rating'], kde=False)
    plt.title(i)
    plt.show()

print("\n Printing the count of generes \n",data["Genres"].value_counts())


print("\n Printing the count of Average User Rating \n",data["Average User Rating"].value_counts())

a = [i[1] for i in list(zip(data["Average User Rating"], data["Genres"])) if i[0]==5.0]
a = pd.Series(a)
a.value_counts()
