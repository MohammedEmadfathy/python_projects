#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import necessary lib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly as py
import plotly.graph_objs as go
import os
py.offline.init_notebook_mode(connected = True)
#print(os.listdir("../input"))
import datetime as dt
import missingno as msno
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# import machine learning module
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation


# In[6]:


df = pd.read_csv("netflix_titles.csv",encoding='latin1')
df.head()


# In[7]:


df = df[df.columns[:12]]
df.head()


# In[8]:


df.info()


# In[9]:


# Check for missing values
df.isnull().sum()


# In[10]:


# Replacments

df['country'] = df['country'].fillna(df['country'].mode()[0])


df['cast'].replace(np.nan, 'No Data',inplace  = True)
df['director'].replace(np.nan, 'No Data',inplace  = True)

# Drops

df.dropna(inplace=True)

# Drop Duplicates

df.drop_duplicates(inplace= True)


# In[11]:


# We need to use the strip module first because some values in this dataset still contain spaces at the beginning or end of string.
df["date_added"] = df["date_added"].str.strip()

# convert dtype to datetime 
df["date_added"] = pd.to_datetime(df['date_added'])

# extract month and year
df['month_added']=df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year


# In[12]:


df.head(5)


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[15]:


# Let's look at Type.
# Create our pie chart with labels

df["type"].value_counts().plot.pie(autopct='%1.2f%%',explode=[0,0.08], shadow = True)


# In[16]:


country_counts = df['country'].value_counts().head(10)  # Top 10 countries

# Create the bar chart
plt.figure(figsize=(10, 6))  # Adjust figure size as desired
bars = plt.bar(country_counts.index, country_counts.values)

# Add count values on top of bars
for bar, count in zip(bars, country_counts.values):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, count, ha='center', va='bottom')

# Highlight top 3 countries
plt.bar(country_counts.index[:3], country_counts.values[:3], color='red')  # Adjust color

# Customize the plot
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Top 10 Countries (Top 3 Highlighted)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

plt.show()


# In[18]:


# Count movies and TV shows per country
movie_counts_country = df[df['type'] == 'Movie']['country'].value_counts()
tv_show_counts_country = df[df['type'] == 'TV Show']['country'].value_counts()

# Combine counts into a single DataFrame with total 
df_counts = pd.DataFrame({'Movie': movie_counts_country, 'TV Show': tv_show_counts_country})
df_counts['total_by_country'] = df_counts.sum(axis=1)

# Sort by total count in descending order and select top 10
top_10_counts = df_counts.sort_values(by='total_by_country', ascending=False).head(10)

# Print the top 10 countries with movie, TV show, and total counts
print(top_10_counts)


# In[21]:


rows, cols = 2, 5
fig, axes = plt.subplots(rows, cols, figsize=(16, 6))  

# Counter to keep track of subplot position
counter = 0

# Loop through each row (country) in the DataFrame
for country, row in top_10_counts.iterrows():
  # Extract movie, tv show, and total counts
  movie_count = row['Movie']
  tv_show_count = row['TV Show']
  total_count = row['total_by_country']

  # Create labels for pie chart slices
  labels = ['Movie', 'TV Show']
 # Create pie chart slice sizes
  sizes = [movie_count, tv_show_count]

  # Select the current subplot based on counter
  ax = axes[counter // cols, counter % cols]

  # Create a pie chart on the selected subplot
  ax.pie(sizes, labels=labels, autopct="%1.1f%%", explode = [0,0.08], shadow = True)
  ax.set_title(country)

  # Increase counter for next subplot position
  counter += 1


# In[22]:


# Count movies and TV shows per year_added
movie_counts_year = df[df['type'] == 'Movie']['year_added'].value_counts()
tv_show_counts_year = df[df['type'] == 'TV Show']['year_added'].value_counts()

# Combine counts into a single DataFrame with total (use add with fill_value=0 for missing values)
df_counts = pd.DataFrame({'Movie': movie_counts_year, 'TV Show': tv_show_counts_year})
df_counts['total_by_year'] = df_counts.sum(axis=1)

# Sort by total count in descending order
rating_agg = df_counts.sort_values(by='year_added', ascending=False)
print(rating_agg)


# In[23]:


#IThink we should drop year 2024, because data maybe incomplete 
df = df[df['year_added'] != 2024]
df.info()


# In[24]:


df_grouped = df.groupby(['year_added', 'type'])['type'].count().unstack(fill_value=0)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 9))
df_grouped.plot(kind='area', stacked=False, ax=ax)
ax.set_title('Cumulative Movies and TV Shows Added per Year')
ax.set_xlabel('Year Added',fontsize = 14)
ax.set_ylabel('Number of Movies/TV Shows', fontsize = 14)
ax.legend(title='Type')

plt.xticks(df['year_added'])
plt.tight_layout()


# In[25]:


# Count movies and TV shows per month_added
movie_counts_month = df[df['type'] == 'Movie']['month_added'].value_counts()
tv_show_counts_month = df[df['type'] == 'TV Show']['month_added'].value_counts()

# Combine counts into a single DataFrame with total (use add with fill_value=0 for missing values)
df_counts = pd.DataFrame({'Movie': movie_counts_month, 'TV Show': tv_show_counts_month})
df_counts['total_by_month'] = df_counts.sum(axis=1)

# Sort by total count in descending order
month_agg = df_counts.sort_values(by='total_by_month', ascending=False)

print(month_agg)


# In[26]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

df['month_added'] = pd.Categorical(df['month_added'], categories=month_order, ordered=True)

df_grouped = df.groupby(['month_added', 'type'])['type'].count().unstack(fill_value=0)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 9))
df_grouped.plot(kind='area', stacked=False, ax=ax)
ax.set_title('Content added by month [Cumulative Total]')
ax.set_xlabel('Month Added',fontsize = 14)
ax.set_ylabel('Number of Movies/TV Shows', fontsize = 14)
ax.legend(title='Type')
plt.xticks(range(len(month_order)), month_order)

plt.tight_layout()
plt.show()


# In[27]:


# Rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='rating', palette='magma', order=df['rating'].value_counts().index)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.xticks(rotation=90)
plt.show()


# In[33]:


# What is the distribution of content ratings on Netflix? How does content rating vary between different countries or regions?
df.rating.value_counts().sort_values().plot(kind='barh')



# In[35]:


from collections import Counter

# Splitting and counting genres
genres = df['listed_in'].apply(lambda x: x.split(','))
genres = [genre.strip() for sublist in genres for genre in sublist]
genre_counts = Counter(genres)

plt.figure(figsize=(8, 12))
plt.barh(list(genre_counts.keys()), list(genre_counts.values()), color='salmon')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.title('Distribution of Genres')
plt.show()


# In[36]:


director = df.director.value_counts().nlargest(10).to_frame().reset_index()
director


# In[ ]:




