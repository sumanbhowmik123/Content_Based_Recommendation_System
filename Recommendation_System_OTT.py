#!/usr/bin/env python
# coding: utf-8

# In[214]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")


# In[215]:


df1=pd.read_csv('MoviesOnOTT.csv')
df1.head()


# In[216]:


df1.shape


# In[217]:


#Dropping Column Type as it has not variance 
df1['Type'].value_counts()


# In[218]:


df1.drop('Type',axis=1,inplace=True)


# In[219]:


df1.columns


# In[220]:


#Checking which columns have a lot of Nan values so that they can be dropped.
df1.isnull().sum()


# ##### Here the features Rotten Tomatoes and Age have 69.19% and 56.99% data missing respectively. Hence dropping them for better model accuracy. 

# In[221]:


#Dropping columns 
df1.drop(['Rotten Tomatoes','Age'],axis=1,inplace=True)


# In[222]:


df1.info()


# In[223]:


#Checking the anomaly for Netflix values_counts
df1['Netflix'].value_counts()


# In[224]:


#Rectifying the anomaly by correcting the value 
df1['Netflix']=df1['Netflix'].replace({'1.01','1.00'})


# In[225]:


#Rectifying the format for Netflix value
df1['Netflix']=df1['Netflix'].astype(int)


# In[226]:


#Creating One column for all the OTT platforms 
df1['Netflix']=df1["Netflix"].replace({1:'Netflix'}).astype(str)
df1['Netflix']=df1['Netflix'].replace({0:np.nan})

df1['Disney+']=df1["Disney+"].replace({1:'Disney_Plus'}).astype(str)
df1['Disney+']=df1["Disney+"].replace({0:np.nan})

df1['Hulu']=df1["Hulu"].replace({1:'Hulu'}).astype(str)
df1['Hulu']=df1["Hulu"].replace({0:np.nan})

df1['Prime Video']=df1["Prime Video"].replace({1:'Prime_Video'}).astype(str)
df1['Prime Video']=df1["Prime Video"].replace({0:np.nan})

df1['OTT_Platform']=df1['Netflix'].str.cat(df1[['Disney+', 'Hulu','Prime Video']], sep=' ')#Creating the Column using the other Columns using a string concatenate


# In[227]:


# Replacing the 0s to get just the names of the platforms.
a=[]
for i in df1['OTT_Platform'].values:
    a.append(i.replace('0','').strip().split())
    


# In[228]:


df1.drop('OTT_Platform',axis=1,inplace=True)


# In[229]:


#Assigning and creating the new OTT_Platform column
df1['OTT_Platform']=a


# In[230]:


def clean2(x):
    return ','.join(x)


# In[231]:


df1['OTT_Platform']=df1['OTT_Platform'].apply(clean2)


# In[232]:


#Imputattion of missing values 

df1.isnull().sum()


# #### IMDb

# In[233]:


#ImDb imputation with median
df1['IMDb'].fillna(df1['IMDb'].median(),axis=0,inplace=True)


# #### Genres 

# In[234]:


df1['Genres'].fillna('Not-Mentioned',axis=0,inplace=True)


# #### Runtime

# In[235]:


df1['Runtime'].fillna(df1['Runtime'].median(),axis=0,inplace=True)


# #### Language

# In[236]:


df1['Language'].fillna('English',axis=0,inplace=True)


# #### Directors

# In[237]:


df1["Directors"].fillna(df1['Directors'].mode()[0],axis=0,inplace=True)


# #### Country

# In[238]:


df1["Country"].fillna(df1['Country'].mode()[0],axis=0,inplace=True)


# ### Business Insights 

# In[239]:


#Checking the number of shows on different types of OTT platforms 
figure, axes = plt.subplots(nrows=2, ncols=2)

plt.title('Netflix')
sns.countplot(df1['Netflix'],ax=axes[0,0])


plt.title('Disney+')
sns.countplot(df1['Disney+'],ax=axes[0,1])


plt.title('Hulu')
sns.countplot(df1['Hulu'],ax=axes[1,0])

plt.title('Prime Video')
sns.countplot(df1['Prime Video'],ax=axes[1,1])


plt.tight_layout()


# ##### As we can infer the maximum number of shows for the data is streamed on Prime Videos followed by Netflix

# In[240]:


#Checking the Distribution of IMDb ratings 
plt.figure(figsize=(8,5))
sns.distplot(df1['IMDb'],bins=30)
plt.grid()
plt.tight_layout()


# #### The distribution seems to follow a Gaussian Distribution with bit of left-skewed.
# 

# In[241]:


def ranged(i):
    if(i<=5):
        return '0-5'
    elif (i>5) and (i<=6):
        return '5-6'
    elif (i>6) and (i<=7):
        return '6-7'
    elif (i>7) and (i<=8):
        return '7-8'
    else:
        return '8-10'


# In[242]:


#Creating and applying a Ranged IMDb to all the IMDb ratings 
df1['IMDb_ranged']=df1['IMDb'].apply(ranged)


# In[243]:


#Checking the balance in IMDb
plt.figure(figsize=(7,4))
sns.countplot(df1['IMDb_ranged'])
plt.grid()
plt.tight_layout()
plt.show()


# In[244]:


#To get the top 10 Directors based on their IMDb ratings 
df1.groupby('Directors').agg({'IMDb':'mean','Title':'count'}).sort_values(by=['Title','IMDb'],ascending=False).head(10)


# ## Genres
# #### H0 : The Genres donot have a significance on the IMDb rating 
# #### H1 : The Genres have a significance on the IMDb rating

# In[245]:


mod=ols('IMDb~Genres',data=df1).fit()


# In[246]:


aov=sm.stats.anova_lm(mod,type=2)


# In[247]:


print(aov)


# #### We reject the Null Hypothesis that stated that Genres doesnot play any significance over IMDb rating 
# #### Accepting H1: Genres has a significance over IMDb rating 

# ## Language
# #### H0 : The Language does not have a significance on the IMDb rating 
# #### H1 : The Language has a significance on the IMDb rating

# In[248]:


mod1=ols('IMDb~Language',data=df1).fit()


# In[249]:


aov1=sm.stats.anova_lm(mod1)


# In[250]:


print(aov1)


# #### We reject the Null Hypothesis that stated that Languages doesnot play any significance over IMDb rating 
# #### Accepting H1: Languages has a significance over IMDb rating 

# ## Recommendation System

# In[251]:


df1=df1.head(3000) #MemoryError due to Large dimension in the count_matrix during Cosine Similarity thus the data has been set to a subset


# #### Cleaning the features which would be used for the recommendation engine and making them ready for sending them into a CountVectorizer

# In[252]:


#Features are-
#Directors
#Genres
#Country
#Language
#IMDb
#Runtime


# In[253]:


#Creating a function to clean all the features required for the recommendation
def clean(x,n):
    #converting strings to lower, replacing the space and splitting the elements with ','
    x=x.lower().replace(' ','').split(',')
    
    #Using a condition to take in maximum number of elements from each feature
    return x[:n] if(len(x)>=1) else x 


# In[254]:


#Cleaned data and took 3 Director Names at max 

df1['Directors']=df1['Directors'].apply(lambda x:clean(x,3)) 


# In[255]:


#Cleaned data and took 4 Genres at max 

df1['Genres']=df1['Genres'].apply(lambda x:clean(x,4))


# In[256]:


#Cleaned data and took 4 Countries at max 

df1['Country']=df1['Country'].apply(lambda x:clean(x,4))


# In[257]:


#Cleaned data and took 4 Languages at max 

df1['Language']=df1['Language'].apply(lambda x:clean(x,3))


# In[258]:


#To scale the continuous variables used MinMaxScaling which will allow them to be in a range of 0-1 
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
df1['IMDb1']=minmax.fit_transform(df1[['IMDb']])
df1['Runtime1']=minmax.fit_transform(df1[['Runtime']])


# In[259]:


#Converting the datatype of numerical features so that they can be added to the CountVectorizer
df1['IMDb1']=df1['IMDb1'].astype(str)
df1['Runtime1']=df1['Runtime1'].astype(str)


# In[260]:


#Combining all the Features for the recommendation into a single column
def combined_features(x):
    return ' '.join(x['Directors'])+' '+' '.join(x['Genres'])+' '+' '.join(x['IMDb1']+' '+' '.join(x['Country'])+' '+' '.join(x['Language'])+' '+' '.join(x['Runtime1']))


# In[261]:


df1['Combined_features']=df1.apply(combined_features,axis=1)


# In[262]:


from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer(stop_words='english')


# In[263]:


#Creating the CountVectorized Matrix which will be used for find the Cosine Similarities
count_matrix = count_vec.fit_transform(df1['Combined_features']) 


# In[264]:


count_tokens=count_vec.get_feature_names()


# In[265]:


count_matrix.shape 


# In[266]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[267]:


#cosine_sim[1]


# In[268]:


indices=pd.Series(df1.index, index=df1['Title']) #Creating a series out of the Title and setting an index to them 


# In[269]:


#indices


# In[270]:



df1['Directors']=df1['Directors'].apply(clean2)

df1['Genres']=df1['Genres'].apply(clean2)

df1['Country']=df1['Country'].apply(clean2)

df1['Language']=df1['Language'].apply(clean2)


# In[271]:


a=list(df1['Title'].unique()) #Created a List of the movie names to filter out Unknown Movie names provided. 


# In[272]:


def get_recommendations(title, cosine_sim=cosine_sim,a=a):
    if title in a:
        #Creating a id for title 
        idx = indices[title]  
        
        # Getting the similarity scores mapped to indices using enumerate and creating a index value
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sorting the similarity scores by reverse=True
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 
        
        # Recommending the 8 movies excluding the one searched for 
        sim_scores = sim_scores[1:9] 
        
        #Looping through the list of similarity scores which are sorted and getting them in a variable 
        movie_indices = [i[0] for i in sim_scores]
        
        #Getting the Title , OTT_Platform and the IMDb rating of the movies which will be recommended
        return df1[['Title','OTT_Platform','IMDb']].iloc[movie_indices]
    else:
        
        #If the Movie Title searched for is not in the list of the Movie Titles passed in 'a'
        print('No Movies Found')


# In[273]:


get_recommendations("Inception")


# In[ ]:





# In[ ]:




