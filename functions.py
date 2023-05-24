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

def get_alcohol_quality(train):
    '''
    Input:
    train df
    Output:
    boxplot of quality and alcohol
    '''
    plt.figure(figsize=(12,6))
    sns.boxplot(data=train, x='quality', y='alcohol', palette='Set1')
    plt.title('Wine by Quality and Alcohol')
    plt.ylabel('Alcohol Vol %')
    plt.xlabel('Quality Rating')
    plt.show()

def get_va_quality(train):
    '''
    Input:
    train df
    Output:
    boxplot of quality and volatile acidity
    '''
    plt.figure(figsize=(12,6))
    sns.boxplot(data=train, x='quality', y='volatile_acidity', palette='Set1')
    plt.title('Wine by Quality and Volatile Acidity')
    plt.ylabel('Volatile Acidity (g/L)')
    plt.xlabel('Quality Rating')
    plt.show()

def get_sulphate_quality(train):
    '''
    Input:
    train df
    Output:
    stripplot of quality and sulphate
    '''
    plt.figure(figsize=(12,6))
    sns.stripplot(data=train, x='quality', y='sulphates')
    plt.title('Wine by Quality and Sulphates')
    plt.ylabel('Sulphates (g/L)')
    plt.xlabel('Quality Rating')
    plt.show()

def get_ca_quality(train):
    '''
    Input:
    train df
    Output:
    barplot of quality and citric acid
    '''
    plt.figure(figsize=(12,6))
    sns.barplot(data=train, x='quality', y='citric_acid', palette='Set1')
    plt.title('Wine by Quality and Citric Acid')
    plt.ylabel('Citric Acid (g/L)')
    plt.xlabel('Quality Rating')
    plt.show()





###### STATS ########


def run_volatile_acidity_ttest(data):
    '''
    runs a Ttest for volatile_acidity vs quality
    '''
    x = data['volatile_acidity']
    y = data['quality']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results


def run_alcohol_ttest(data):
    '''
    runs a Ttest for alcohol vs quality
    '''
    x = data['alcohol']
    y = data['quality']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_citric_acid_ttest(data):
    '''
    runs a Ttest for citric acid vs quality
    '''
    x = data['citric_acid']
    y = data['quality']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_sulphates_ttest(data):
    '''
    runs a Ttest for sulphates vs quality
    '''
    x = data['sulphates']
    y = data['quality']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results



######## modeling ###########

def scale_data(train, validate, test, columns):
    """
    Scale the selected columns in the train, validate, and test data.
    Args:
        train (pd.DataFrame): Training data.
        validate (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        columns (list): List of column names to scale.
    Returns:
        tuple: Scaled data as (X_train_scaled, X_validate_scaled, X_test_scaled).
    """
    # create X & y version of train, where y is a series with just the target variable and X are all the features.
    X_train = train.drop(['quality','total_sulfur_dioxide','wine_type'], axis=1)
    y_train = train['quality']
    X_validate = validate.drop(['quality','total_sulfur_dioxide','wine_type'], axis=1)
    y_validate = validate['quality']
    X_test = test.drop(['quality','total_sulfur_dioxide','wine_type'], axis=1)
    y_test = test['quality']
    # Create a scaler object
    scaler = MinMaxScaler()
    # Fit the scaler on the training data for the selected columns
    scaler.fit(X_train[columns])
    # Apply scaling to the selected columns in all data splits
    X_train_scaled = X_train.copy()
    X_train_scaled[columns] = scaler.transform(X_train[columns])

    X_validate_scaled = X_validate.copy()
    X_validate_scaled[columns] = scaler.transform(X_validate[columns])

    X_test_scaled = X_test.copy()
    X_test_scaled[columns] = scaler.transform(X_test[columns])
    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test

def get_baseline(y_train):
    '''
    this function returns a baseline for accuracy
    '''
    baseline_prediction = y_train.median()
    # Predict the majority class in the training set
    baseline_pred = [baseline_prediction] * len(y_train)
    accuracy = accuracy_score(y_train, baseline_pred)
    baseline_results = {'Baseline': [baseline_prediction],'Metric': ['Accuracy'], 'Score': [accuracy]}
    baseline_df = pd.DataFrame(data=baseline_results)
    return baseline_df  



def create_models(seed=123):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    This includes best fit hyperparamaenters                
    '''
    models = []
    models.append(('k_nearest_neighbors', KNeighborsClassifier(n_neighbors=100)))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=3,min_samples_split=4,random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(max_depth=3,random_state=seed)))
    return models

def get_models(train, validate, test):
    # create models list
    models = create_models(seed=123)
    X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test = scale_data(train, validate, test, ['alcohol', 'volatile_acidity','sulphates','citric_acid','free_sulfur_dioxide','ph','fixed_acidity','residual_sugar','white','chlorides','density'])
    # initialize results dataframe
    results = pd.DataFrame(columns=['model', 'set', 'accuracy', 'recall'])
    
    # loop through models and fit/predict on train and validate sets
    for name, model in models:
        # fit the model with the training data
        model.fit(X_train_scaled, y_train)
        
        # make predictions with the training data
        train_predictions = model.predict(X_train_scaled)
        
        # calculate training accuracy 
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        
        # make predictions with the validation data
        val_predictions = model.predict(X_validate_scaled)
        
        # calculate validation accuracy and recall
        val_accuracy = accuracy_score(y_validate, val_predictions)
        
        
        # append results to dataframe
        results = results.append({'model': name, 'set': 'train', 'accuracy': train_accuracy}, ignore_index=True)
        results = results.append({'model': name, 'set': 'validate', 'accuracy': val_accuracy}, ignore_index=True)
        '''
        this section left in case I want to return to printed format rather than data frame
        # print classifier accuracy and recall
        print('Classifier: {}, Train Accuracy: {}, Validation Accuracy: {}'.format(name, train_accuracy, val_accuracy))
        '''
    return results