import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import os
 

####################### Imports ############################

def check_file_exists(fn, query, url):
    '''
    this function checks to see if the .csv file already exists. If yes it reads it
    '''
    if os.path.isfile(fn):
        print('csv file found and loaded\n')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('first create combined_wine.csv')
        return df 
    

    
    
    
    
