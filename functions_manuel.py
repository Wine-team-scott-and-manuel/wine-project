######IMPORTS#####

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Basics:
import pandas as pd
import numpy as np
import math
import numpy as np
import scipy.stats as stats
import os

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns


# Sklearn stuff:
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import mutual_info_score

from sklearn.model_selection import train_test_split

## Regression Models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

## Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

## local
import wrangle



###### VISUALIZATIONS ########







###### STATS ########







