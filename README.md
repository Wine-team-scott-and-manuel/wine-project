# Wine-Clustering-Project
### Manuel Salazar and Scott Barnett
## Project description with goals
### Description
* We want to be able to predict the quality score of wine based on the features given in the data set

### Goals¶
* Construct a ML model that predicts wine quality 
* Find key drivers of wine quality 
* Deliver a report to the data science team 
* Deliver a presentation of findings to the Data Science team

## Initial hypotheses and/or questions you have of the data, ideas
There should be some combination of features that can be used to build a predictive model for wine quality
* 1. Is alcohol content predictive of quality?
* 2. Is the volatile acidity level predictive of quality?
* 3. Are sulphates predictive of quality?
* 4. Is citric acid predictive of quality? 
*****************************************
## Project Plan 
* Data acquired from https://data.world/food/wine-quality
    * Seperate red and white wine .csv files were imported
    * Files were concated with a column added to distinguish red from white wines 
* Prepare data
    * Change column names to remove spaces
    * Converted total and free sulfur dioxide from mg/L to g/L to match the rest of the data measurements
    * Created dummy column for wine type
    * There were no null or missing values
    * Values af 0 were left assuming that 0 is a valid possibility 
    * Created univariate histogram for visual first glance at data
    * Outliers were not addressed in this itteration
    * Split data into **train**, **validate**, **test**       
## Explore data in search of drivers of wine quality
* Answer the following initial questions
    * 1. Is alcohol content predictive of quality?
    * 2. Is the volatile acidity level predictive of quality?
    * 3. Are sulphates predictive of quality?
    * 4. Is citric acid predictive of quality? 
    * 5. Are there any clusters that would be useful for our model?
* Develop a model to predict wine quality
    * Run the models with and without clusters 
    * Observe impact clusters have on model performance
    * Select best model
* Draw conclusions

## Data Dictionary
| Feature | Definition (measurement)|
|:--------|:-----------|
|Fixed Acidity| The fixed amount of tartaric acid. (g/L)|
|Volatile Acidity| A wine's acetic acid; (High Volatility = High Vinegar-like smell). (g/L)|
|Citric Acid| The amount of citric acid; (Raises acidity, Lowers shelf-life). (g/L)|
|Residual Sugar| Leftover sugars after fermentation. (g/L)|
|Chlorides| Increases sodium levels; (Affects color, clarity, flavor, aroma). (g/L)|
|Free Sulfur Dioxide| Related to pH. Determines how much SO2 is available. (Increases shelf-life, decreases palatability). (mg/L)|
|Total Sulfur Dioxide| Summation of free and bound SO2. (Limited to 350ppm: 0-150, low-processed, 150+ highly processed). (mg/L)|
|Density| Between 1.08 and 1.09. (Insight into fermentation process of yeast growth). (g/L)|
|pH| 2.5: more acidic - 4.5: less acidic (range)|
|Sulphates| Added to stop fermentation (Preservative) (g/L)|
|Alcohol| Related to Residual Sugars. By-product of fermentation process (vol%)|
|Quality| Score assigned between 0 and 10; 0=low, 10=best|
|Wine type| Classifies color of wine ; Red or White|

## Steps to Reproduce
* 1. Data is collected from Data.World Wine Quality Dataset
    * Download the red and white wine .csv's
    * Combine the two and add a column to identify red or white wine
    * Save as 'combined_wine.csv' to use in repo
* 2. Clone this repo.
* 3. Put the data in the file containing the cloned repo.
* 4. Run notebook.

## Takeaways and Conclusions
* The four features we evaluated closely: alcohol, volatile acidity, citric acid and sulphates; were all closely related to quality scores 
* Only total_sulfur_dioxide needed to be removed from the data set due to it's close relationship to free_sulfur_dioxide
* There was no real opportunity for feature engineering
    * We did some required work for the machine learning models
    * We had virtualy no success finding useful clusters
* Our best model was KNN ran **without** clusters
    * It returned an Accuracy score of 56.08% out performing the baseline by more than 12%
    * Forcing the cluster for the sake of comparison caused the model to perform worse

# Recomendations
* The model provided can be used to create an estimated quality score
* However, the human sommelier still has a superior nose for this process
# Next Steps
* If provided more time to work on the project we would want to explore taking sommelier classes to see if there is data that could be garnered for the model