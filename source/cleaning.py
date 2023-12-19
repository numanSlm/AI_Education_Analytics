# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

# Creating DataFrames from csv

fer_df = pd.read_csv("./fer2013/fer2013.csv")
fer_new_df = pd.read_csv("./fer2013/fer2013new.csv")

print("fer_df shape is {}, and fer_new_df shape is {}\n".format(fer_df.shape, fer_new_df.shape))


# Mergint to daraframes
df = pd.concat([fer_df,fer_new_df],axis=1)
df.drop(["Usage",'Image name'],axis=1,inplace=True)

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

df['emotion'] = df.apply(lambda x: emotions[x['emotion']],axis=True)
sns.countplot(data=df,x='emotion')
plt.savefig("initial_countplot.svg")

# cleaning dataset from Not Face images
NF_unique_values = df.sort_values(by='NF',ascending=False)['NF'].unique()
print("Number of unique values of Not Face class: {}\n".format(NF_unique_values))

for i in NF_unique_values:
    print("number of images with score NF={} is {}".format(i,df[df['NF'] == i].shape[0]))
    
# Saving figures of Not face classes
img_visualizer(df,list(df[df['NF'] == 10].head(25).index), 5,5,save_name='NF10.svg',show_label=False)
img_visualizer(df,list(df[df['NF'] == 4].index), 1,2,save_name='NF4.svg',show_label=False)
img_visualizer(df,list(df[df['NF'] == 2].index), 2,2,save_name='NF2.svg',show_label=False)
img_visualizer(df,list(df[df['NF'] == 1].head(25).index), 5,5,save_name='NF2.svg',show_label=False)

df = df[df["NF"] == 0]
print("Shape of DataFrame after removing Not Face labeled Images: {}".format(df.shape))
# cleaning dataset from Uknown images
unknown_unique_values = df.sort_values(by='unknown',ascending=False)['unknown'].unique()
print("Number of unique values of Not Face class: {}\n".format(unknown_unique_values))

for i in unknown_unique_values:
    print("number of images with score unknown={} is {}".format(i,df[df['unknown'] == i].shape[0]))
    
# Saving figures of Not face classes
img_visualizer(df,list(df[df['unknown'] >= 7].index), 2,3,save_name='uk87.svg',show_label=False)
img_visualizer(df,list(df[df['unknown'] == 6].head(16).index), 4,4,save_name='uk6.svg',show_label=False)
img_visualizer(df,list(df[df['unknown'] == 5].head(25).index), 5,5,save_name='uk5.svg',show_label=False)

df = df[df["unknown"] < 5]

#extracting new lalels from the scores of the dataframe
new_ome = df.apply(lambda x:new_emotions(x['neutral'],x['happiness'],x['surprise'],x['sadness'],
                                         x['anger'],x['disgust'],x['fear']),axis=1)
df['new_emotions'] = new_ome

# saving
ndf = df[df['new_emotions'] != 'other']
ndf = ndf[['new_emotions', 'pixels']]
ndf['emotions'] = ndf['new_emotions']

ndf.to_csv("Cleaned_data.csv")
df = pd.read_csv("Cleaned_data.csv")


for i in ['nautral','angry','focused','bored']:
    img_visualizer(df,list(df[df['new_emotions'] == i].head(25).index), 5,5,save_name='{}_sample.svg'.format(i),show_label=False)
    print("number of samples for {}: {}".format(i,df[df['new_emotions'] == i].shape[0]))

sns.countplot(data=df,x='emotions')
plt.savefig("why.svg") 