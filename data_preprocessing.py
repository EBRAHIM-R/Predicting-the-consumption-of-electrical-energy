#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)
from sklearn. compose import ColumnTransformer 
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#convert timestamp to (Universal Time Coordinated) global time
STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc") 
STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")

# loading dataset
de_load = pd.read_csv("dataset/de.csv")

# show data set
de_load.head()


# In[3]:


# indexing data set with "start" column and remove "end" column
de_load = de_load.drop(columns="end").set_index("start") 
de_load.index = pd.to_datetime(de_load.index)

# raname index to become "time"
de_load.index.name = "time"

# grouping dataset each hour
de_load = de_load.groupby(pd.Grouper(freq="h")).mean()

# selecting dataset betwen STUDY_START_DATE and STUDY_END_DATE
de_load = de_load.loc[
    (de_load.index >= STUDY_START_DATE) & (de_load.index <= STUDY_END_DATE), :
]

# show data set
de_load.head()


# In[4]:


# define split_train_test function to split dataset to train dataset and test dataset depending on "split_time"
def split_train_test(df, split_time):
    df_train = df.loc[df.index < split_time]
    df_test = df.loc[df.index >= split_time]
    return df_train, df_test

# Apply split_train_test on dataset
df_train, df_test = split_train_test(
    de_load, pd.Timestamp("2019-02-01", tz="utc")
)


# In[5]:


#The following features are used for create models :
#time features: month, weekday and hour
#national holiday features, as a boolean time series
#lag features: load data with a lag values ranging from 24 to 72 hours

# create function to add time features 
def add_time_features(df):
    
    # convert time to Central European Time
    cet_index = df.index.tz_convert("CET")
    
    # add "month","weekday" and "hour" features 
    df["month"] = cet_index.month
    df["weekday"] = cet_index.weekday 
    df["hour"] = cet_index.hour
    return df

# create function to add weekend days
def add_holiday_features(df):
    
    # get all weekend days in Germany in this years
    de_holidays = holidays.Germany(years = [2015, 2016, 2017, 2018, 2019]) 
    
    # create a series of dataset
    cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index) 
    
    # "apply function" take one parameter and applies that on all values in "cet_dates" series.
    # add feature "holiday" feature
    df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
    
    # convert type to int
    df["holiday"] = df["holiday"].astype(int) 
    return df

#create function to add "lag" features to analyse data in different periods
def add_lag_features(df, col="load"):
    for n_hours in range(24, 73):
        shifted_col = df[col].shift(n_hours, "h") 
        shifted_col = shifted_col.loc[df.index.min(): df.index.max()] 
        label = f"{col}_lag_{n_hours}"
        df[label] = np.nan
        df.loc[shifted_col.index, label] = shifted_col
    return df

# create function to call all previous functions on dataset
def add_all_features(df, target_col="load"):
    df = df.copy()
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, col=target_col)
    return df

# call a last function on training_dataset and remove all "Nan" values from this
df_train = add_all_features(df_train).dropna()
# call a last function on testing_dataset and remove all "Nan" values from this
df_test = add_all_features(df_test).dropna()

# show training_dataset
df_train.head()


# In[6]:


# target is predict "load"
target_col = "load"

# split training_dataset to X_train and y_train
X_train = df_train.drop(columns=target_col)
y_train = df_train.loc[:, target_col]

# split testing_dataset to X_test and y_test
X_test = df_test.drop(columns=target_col)
y_test = df_test.loc[:, target_col]


# In[7]:


# define fit_prep_pipeline to nomalizing dataset
def fit_prep_pipeline(df):
    
    # categorical features
    cat_features = ["month", "weekday", "hour"]  
    
    # binary features
    bool_features = ["holiday"]  
    
    # numerical features
    num_features = [c for c in df.columns
                    if c.startswith("load_lag")]  
    
    # transformation values to become the same scaling
    prep_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_features),
        ("bool", FunctionTransformer(), bool_features),  # identity
        ("num", StandardScaler(), num_features),
    ])
    prep_pipeline = prep_pipeline.fit(df)
    
    # put all new features in "features_names" list
    feature_names = []
    one_hot_tf = prep_pipeline.transformers_[0][1]
    for i, cat_feature in enumerate(cat_features):
        categories = one_hot_tf.categories_[i]
        cat_names = [f"{cat_feature}_{c}" for c in categories]
        feature_names += cat_names
    feature_names += (bool_features + num_features)
    return feature_names, prep_pipeline


# In[8]:


# apply fit_prep_pipeline function to training_dataset and rename it to "X_train_prep"
feature_names, prep_pipeline = fit_prep_pipeline(X_train)
X_train_prep = prep_pipeline.transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names, index=df_train.index)

# apply fit_prep_pipeline function to testing_dataset and rename it to "X_test_prep"
X_test_prep = prep_pipeline.transform(X_test)
X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names, index=df_test.index)

# shape of training_dataset
print("shape of training_dataset is:")
print(X_train_prep.shape)

# shape of testing dataset
print("shape of testing_dataset is:")
print(X_test_prep.shape)


# In[ ]:




