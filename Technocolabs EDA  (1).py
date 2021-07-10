#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[301]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as dt


# # Reading file

# In[302]:


df=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\HistoricalData.csv",encoding='ISO-8859-1')


# In[303]:


df.head()


# In[304]:


df.tail()


# In[305]:


df.dtypes


# # Renaming Columns

# In[306]:


df=df.rename(columns={'Close/Last':'Close'})


# In[307]:


df.head()


# # Converting Data type

# In[308]:


df['Date']=pd.to_datetime(df['Date'],format='%m/%d/%Y')


# In[309]:


df.head()


# In[310]:


df['Close']=df['Close'].replace('[\$]','',regex=True).astype(float)


# In[311]:


df[df.columns[3:]]=df[df.columns[3:]].replace('[\$]','',regex=True).astype(float)


# In[312]:


df


# In[313]:


df.describe()


# # Detecting Outliers

# In[314]:


plt.figure(figsize=(10,5))
sns.distplot(df['Close'])
plt.show()


# In[315]:


sns.boxplot(df["Close"])


# # Handling Outliers using Inter Quartile Range

# In[316]:


Q1=df["Close"].quantile(0.25)
Q3=df["Close"].quantile(0.75)
print(Q1,Q3)
iqr=Q3-Q1
iqr


# In[317]:


upper_limit = Q3 + 1.5 * iqr
lower_limit = Q1 - 1.5 * iqr
print(upper_limit,lower_limit)


# In[318]:


new_df = df[df['Close'] < lower_limit] 
new_df.shape


# In[319]:


new_df = df[df['Close'] > upper_limit] 
new_df.shape


# # Trimming outliers

# In[320]:


new_df = df[df['Close'] < upper_limit] 
new_df.shape


# In[321]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Close'])
plt.subplot(2,2,2)
sns.boxplot(df['Close'])
plt.subplot(2,2,3)
sns.distplot(new_df['Close'])
plt.subplot(2,2,4)
sns.boxplot(new_df['Close'])
plt.show()


# # Capping Outliers

# In[322]:


new_df_cap = df.copy()
new_df_cap['Close'] = np.where(
    new_df_cap['Close'] > upper_limit,upper_limit,
    np.where(new_df_cap['Close'] < lower_limit,lower_limit,
        new_df_cap['Close']
    )
)


# In[323]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Close'])
plt.subplot(2,2,2)
sns.boxplot(df['Close'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['Close'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['Close'])
plt.show()


# # Reading Headlines Dataset

# In[324]:


df15=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2015.csv",encoding='ISO-8859-1')
df16=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2016.csv",encoding='ISO-8859-1')
df17=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2017.csv",encoding='ISO-8859-1')
df18=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2018.csv",encoding='ISO-8859-1')
df19=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2019.csv",encoding='ISO-8859-1')
df20=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2020.csv",encoding='ISO-8859-1')
df21=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\2021.csv",encoding='ISO-8859-1')


# In[325]:


df1=pd.concat([df15,df16,df17,df18,df19,df20,df21])


# In[326]:


df1


# In[327]:


df1.dtypes


# In[328]:


df1['Date']=pd.to_datetime(df1['Date'],format='%Y/%m/%d')


# In[329]:


df1["Headlines"]=df1["Headlines"].astype(str)
df1['Headlines']=df1['Headlines'].apply(lambda x: x.lower())


# # Removing Punctuations and Stopwords

# In[330]:


import string
def punctuation_removal(text):
    li=[char for char in text if char not in string.punctuation]
    clean_str=''.join(li)
    li.clear()
    return clean_str
df1["Headlines"]=df1["Headlines"].astype(str)
df1['Headlines']=df1['Headlines'].apply(punctuation_removal)
df1


# In[331]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
df1['Headlines'] = df1['Headlines'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[332]:


df1


# In[333]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def split_to_lemma(text):
    lemma=WordNetLemmatizer()
    words=word_tokenize(text)
    return ' '.join([lemma.lemmatize(word) for word in words])
df1['Headlines'] = df1['Headlines'].apply(split_to_lemma)


# In[334]:


df1


# # Merging both Datasets

# In[335]:


df=pd.merge(df,df1,on=['Date'],how='outer')


# In[336]:


df


# # Handling Null Values

# In[337]:


df.isnull().sum()


# In[338]:


Close_mean=df["Close"].mean()
Volume_mean=df["Volume"].mean()
Open_mean=df["Open"].mean()
High_mean=df["High"].mean()
Low_mean=df["Low"].mean()


# In[339]:


df['Close'].fillna(value=Close_mean,inplace=True)
df['Volume'].fillna(value=Volume_mean,inplace=True)
df['Open'].fillna(value=Open_mean,inplace=True)
df['High'].fillna(value=High_mean,inplace=True)
df['Low'].fillna(value=Low_mean,inplace=True)


# In[340]:


df=df.dropna()


# In[341]:


df


# In[342]:


hist=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\hist.csv",encoding='ISO-8859-1')
m1=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\m1.csv",encoding='ISO-8859-1')
m2=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\m2.csv",encoding='ISO-8859-1')
s1=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\s1.csv",encoding='ISO-8859-1')
s2=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\s2.csv",encoding='ISO-8859-1')
s3=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\s3.csv",encoding='ISO-8859-1')
s4=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\s4.csv",encoding='ISO-8859-1')
s5=pd.read_csv(r"C:\Users\rmkis\Desktop\Datasets\s5.csv",encoding='ISO-8859-1')


# In[343]:


hist.columns


# In[344]:


hist.rename(columns={'Headlines Securities CIK':'Security CIK','Headlines Securities CUSIP':'Security CUSIP', 'Headlines Securities Symbol':'Security Symbol','Headlines Securities ISIN':'Security ISIN','Headlines Securities Valoren':'Security Valoren','Headlines Securities Name':'Security Name','Headlines Securities Market':'Security Market','Headlines Securities MarketIdentificationCode':'Security MarketIdentificationCode','Headlines Securities MostLiquidExchange':'Security MostLiquidExchange','Headlines Securities CategoryOrIndustry':'Security CategoryOrIndustry'},inplace=True)


# In[345]:


m1.columns


# In[346]:


m1.rename(columns={'Headlines Securities CIK':'Security CIK','Headlines Securities CUSIP':'Security CUSIP', 'Headlines Securities Symbol':'Security Symbol','Headlines Securities ISIN':'Security ISIN','Headlines Securities Valoren':'Security Valoren','Headlines Securities Name':'Security Name','Headlines Securities Market':'Security Market','Headlines Securities MarketIdentificationCode':'Security MarketIdentificationCode','Headlines Securities MostLiquidExchange':'Security MostLiquidExchange','Headlines Securities CategoryOrIndustry':'Security CategoryOrIndustry'},inplace=True)


# In[347]:


m2.columns


# In[348]:


m2.rename(columns={'Headlines Securities CIK':'Security CIK','Headlines Securities CUSIP':'Security CUSIP', 'Headlines Securities Symbol':'Security Symbol','Headlines Securities ISIN':'Security ISIN','Headlines Securities Valoren':'Security Valoren','Headlines Securities Name':'Security Name','Headlines Securities Market':'Security Market','Headlines Securities MarketIdentificationCode':'Security MarketIdentificationCode','Headlines Securities MostLiquidExchange':'Security MostLiquidExchange','Headlines Securities CategoryOrIndustry':'Security CategoryOrIndustry'},inplace=True)


# In[349]:


s1.columns


# In[391]:


data=pd.concat([hist,m1,m2,s1,s2,s3,s4,s5])


# In[392]:


data


# In[393]:


cols=['Outcome','Identity','Delay','Security CIK','Security Valoren','Headlines UTCOffset','Headlines Url','Headlines Images','Headlines PaywallType','Headlines Source','Security MostLiquidExchange','Message','Security CUSIP','Security ISIN','Headlines Time','Security MarketIdentificationCode','Headlines Tags TagType']


# In[394]:


data=data.drop(cols,axis=1)


# In[395]:


data


# In[396]:


data.reset_index(drop=True,inplace=True)


# In[397]:


data.duplicated().sum()


# In[398]:


data.drop_duplicates(inplace=True,ignore_index=True)


# In[399]:


data.shape


# In[400]:


data.dtypes


# In[401]:


data['Headlines Date']=pd.to_datetime(data['Headlines Date'],format='%m/%d/%Y')


# In[402]:


day=data['Headlines Date'].dt.day
sns.distplot(day,bins=31)


# In[403]:


month=data['Headlines Date'].dt.month
sns.distplot(month,bins=12,kde=False)


# In[404]:


data=data[data['Security Symbol']=='AAPL']


# In[405]:


data


# In[406]:


data.reset_index(inplace=True)


# In[407]:


data


# In[408]:


data['Security CategoryOrIndustry'].value_counts()


# In[409]:


data['Security Name'].value_counts()


# In[410]:


data['Security Market'].value_counts()


# In[411]:


data['Headlines Tags TagValues'].value_counts()


# In[412]:


data.drop(['Security CategoryOrIndustry','Headlines Tags TagValues','Security Name','Security Market','Security Symbol'],axis=1,inplace=True)


# In[413]:


data.rename(columns={'Headlines Date':'Date'},inplace=True)


# In[414]:


data.sort_index(axis = 1,inplace=True)


# In[415]:


data.duplicated().sum()


# In[416]:


data['Headlines Summary'].fillna("",inplace=True)


# In[417]:


data


# In[418]:


data['Headlines Summary'][1]


# In[422]:


for row in range(0,len(data.index)):
    data['Headlines'][row]=' '.join(str(i) for i in data.iloc[row,1:3])


# In[423]:


data


# In[424]:


data.drop(['Headlines Summary','Headlines Title','index'],axis=1,inplace=True)


# In[425]:


data


# In[426]:


data['Headlines']=data['Headlines'].apply(punctuation_removal)


# In[427]:


data['Headlines']=data['Headlines'].astype(str)
data['Headlines'] = data['Headlines'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[428]:


data['Headlines'] = data['Headlines'].apply(split_to_lemma)


# In[429]:


final_dataset=pd.merge(df,data,on=['Date'],how='outer')


# In[430]:


final_dataset


# In[433]:


data['Headlines'].isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




