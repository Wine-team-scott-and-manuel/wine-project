import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split

####################### Imports ############################

def get_wine():
    '''
    combined csv needs to be name 'combined_wine.csv' in local drive
    renames columns for redability
    converts mg/L columns to g/L for uniformity
    creates dummy for wine_type
    returns df
    '''
    df = pd.read_csv('combined_wine.csv')
    new_col_name = []
    for col in df.columns:
        new_col_name.append(col.lower().replace(' ', '_'))
    df.columns = new_col_name
    df['total_sulfur_dioxide'] = df.total_sulfur_dioxide / 1000
    df['free_sulfur_dioxide'] = df.free_sulfur_dioxide / 1000
    dummy_df = pd.get_dummies(df['wine_type'], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df
    
def split_data(df):
    '''
    Be sure to code it as train, validate, test = split_data(df)
    take in a DataFrame and return train, validate, and test DataFrames; .
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       )
    #This confirms and Validates my split.
    
    print(f'train -> {train.shape}, {round(train.shape[0]*100 / df.shape[0],2)}%')
    print(f'validate -> {validate.shape},{round(validate.shape[0]*100 / df.shape[0],2)}%')
    print(f'test -> {test.shape}, {round(test.shape[0]*100 / df.shape[0],2)}%')
    
    return train, validate, test    
    
    
    
