![Rossmann Store Sales](https://user-images.githubusercontent.com/91201232/134559863-1b84aae6-b027-4fd0-9bf6-8be122d0ec83.png)
<br>
<br>

## CONTENTS
- [Business](#business)
- [Data](#data)
- [Steps](#Steps)
- [Progress](#Progress)
- [Tools](#tools)
- [Originator](#originator)
<br>

## BUSINESS

**Forecast sales using store, promotion, and competitor data**

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

There are historical sales data for 1,115 Rossmann stores.

Case link: https://www.kaggle.com/c/rossmann-store-sales
<br>
<br>

## DATA
You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.


### Files
The files are located in the `data` directory.

- train.csv - historical data including Sales
- test.csv - historical data excluding Sales
- sample_submission.csv - a sample submission file in the correct format
- store.csv - supplemental information about the stores

### Data Fields

Most of the fields are self-explanatory. The following are descriptions for those that aren't.

- Id - an Id that represents a (Store, Date) duple within the test set
- Store - a unique Id for each store
- Sales - the turnover for any given day (this is what you are predicting)
- Customers - the number of customers on a given day
- Open - an indicator for whether the store was open: 0 = closed, 1 = open
- StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
- SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
- StoreType - differentiates between 4 different store models: a, b, c, d
- Assortment - describes an assortment level: a = basic, b = extra, c = extended
- CompetitionDistance - distance in meters to the nearest competitor store
- CompetitionOpenSince [Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
- Promo - indicates whether a store is running a promo on that day
- Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
- Promo2Since [Year/Week] - describes the year and calendar week when the store started participating in Promo2
- PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
<br>

## STEPS

The method used for the project was CRISP-DM, apply as the steps below:

- [x] **Data Description:** The goal is to use statistical metrics to identify outliers in the business scope and also analyze basic statistical metrics such as: mean, median, maximum, minimum, range, skew, curtosis and standard deviation.

- [x] **Feature Engineering:** The goal of this step is to obtain new attributes based on the original variables, in order to better describe the phenomenon to be modeled.

- [x] **Variable Filtering:** The goal of this step it to filter rows and delete columns that are not relevant for the model or are not part of the business scope.

- [x] **Exploratory Data Analysis (EDA):** The goal of this step is to explore the data to find insights and better understand the impact of variables on model learning.

- [x] **Data Preparation:** The goal of this step is to prepare the data prepare data for application of the machine learning model.

- [x] **Feature Selection:** The goal of this step is to select the better attributes to train the model. It was used Boruta Algorithm to make the selection.

- [x] **Machine Learning Modeling:** The goal of this step is to do the machine learning model training.

- [x] **Hyperparameter Fine Tunning:** The goal of this step is to choose the best values for each of the parameters of the model selected in the previous step.

- [x] **Convert model performance to business values:** The goal of this step is to convert model performance to a business result.

- [x] **Deploy Model to Production:** The goal of this step is to publish the model in a cloud environment so that other people or services can use the results to improve the business decision. The cloud application platform choosed was Heroku.

- [x] **Telegram Bot:** The goal of this step is to create a bot on the telegram app, that make possible to consult the forecast at any time.

<br>
 
## PROGRESS

### Data Description
- Understanding the business
- Method CRISP-DM
- Rename columns for default
- Data Dimensions
- Checks data types
- Check Missing Values (N/A)
- Correcting N/A based on business
- Modifying dates type after correction N/A
- Descriptive Statistical
  - Descriptive Statistical of numerical variable (median, mean, std, skew, kurtosis)
  - Descriptive Statistical of categorical variable (boxplot)

### Feature Engineering
- Creating Hypothesis Mind Map (Insights)
- Hypothesis Creation
- Final List of Hypotheses
- Feature Engineering application
- Deriving the variables

### Variable Filtering
- Line filtering (Restrictions '0 = no sales')
- Selection of columns (Delet restrictions)

### Exploratory Data Analysis (EDA)
- Differentiation between univariate, bivariate and multivariate analysis
- Exploratory data analysis on numerical and categorical Univariate Variables
- Exploratory data analysis on Bivariate Variables (Hypothesis validation)
- Validation of business hypotheses using 'heatmap', 'regplot' and 'barplot' graphics
- Multivariate analysis on numerical and categorical variables using heatmap for numerical variables and Cram??r V calculation for categorical variables.

### Data Preparation
- When necessary apply Normalization
- Apply Rescaling ( RobustScaling for variables with higher outliers and MinMaxScaling for variables with lower outliers)
- Apply Encoding on Categorical Variable - LabelEncoding, One Hot Encoding and Ordinal Encoding
- Implementation of the Nature Transformation of variables with cyclical characteristics (day, month, week of the year, day of the week) using trigonometric techniques (sin and cos)

### Feature Selection
- Objective of understand the relevance of the variables to the model.
- Collinear Variables - Using one of the 3 variable selection methods, these 'collenear variables' are removed for explaining the same thing.

  Variable Selection Methods:
    - Filter Methods
    - Embedded Methods
    - Wrapper Methods (Feature Selection - Boruta)

### Machine Learning Modelling
- Definition of the five best models for regression given the problem of forecasting sales in stores
  - Average Model
  - Linear Regression
  - Linear Regression Regularized - Lasso
  - Random Forest Regressor
  - XGBoost Regressor
- Comparing the Machine Learning Modelling
- Performing Cross Validation on the data
- Comparison of the performance of models after cross validation

### Hyperparameter Fine Tuning
- Selection of the best parameters for the model to be trained.
- Random Search application to find parameter values (Grid Search, Random Search and Bayesian Search)
- Although Random Forest expresses better values for this project, we will proceed with XGBoost to delve into Hyperparameter Fine Tuning. After several iterations, we arrived at the optimal training values for XGBoost:

```python

param_tuned = {
    'n_estimators': 1500, 
    'eta': 0.03,
    'max_depth': 9,
    'subsample': 0.1,
    'colsample_bytree': 0.3,
    'min_child_weight': 8
        }

MAX_EVAL = 2

```

### Convert model performance to business values

```python
## Business Performance
# Sum of predictions
dataset91 = dataset9[['store', 'predictions']].groupby( 'store' ).sum().reset_index()

# MAE and MAPE
dataset9_aux1 = dataset9[['store', 'sales', 'predictions']].groupby( 'store' ).apply( 
lambda x: mean_absolute_error( x['sales'], x['predictions'] ) ).reset_index().rename( columns={0: 'MAE'} )
dataset9_aux2 = dataset9[['store', 'sales', 'predictions']].groupby( 'store' ).apply( 
lambda x: mean_absolute_percentage_error( x['sales'], x['predictions'] ) ).reset_index().rename( columns={0: 'MAPE'} )

# Merge
dataset9_aux3 = pd.merge( dataset9_aux1, dataset9_aux2, how='inner', on='store')
dataset92 = pd.merge( dataset91 , dataset9_aux3, how='inner', on='store')

# Scenarios
dataset92['worst_scenario'] = dataset92['predictions'] - dataset92['MAE']
dataset92['best_scenario'] = dataset92['predictions'] + dataset92['MAE']

# Order Columns
dataset92 = dataset92[[ 'store', 'predictions', 'worst_scenario', 'best_scenario', 'MAE', 'MAPE' ]]

```
<p align="center">
  <img src="https://github.com/IzabellaSouza/Rossmann-Store-Sales/blob/main/img/Business_Performance.PNG">
</p>

<p align="center">
<img src="https://github.com/IzabellaSouza/Rossmann-Store-Sales/blob/main/img/total_performance.PNG">
</p>

```python
## Machine Learning Performance
dataset9['error'] = dataset9['sales'] - dataset9['predictions']
dataset9['error_rate'] = dataset9['predictions'] / dataset9['sales']

plt.figure(figsize=(18, 9))

plt.subplot( 2, 2, 1 )
sns.lineplot( x='date', y='sales', data=dataset9, label='SALES')
sns.lineplot( x='date', y='predictions', data=dataset9, label='PREDICTIONS')

plt.subplot( 2, 2, 2 )
sns.lineplot( x='date', y='error_rate', data=dataset9 )
plt.axhline( 1, linestyle='--' )

plt.subplot( 2, 2, 3 )
sns.distplot( dataset9['error'] )

plt.subplot( 2, 2, 4 )
sns.scatterplot( dataset9['predictions'], dataset9['error'] )
```
<img src="https://github.com/IzabellaSouza/Rossmann-Store-Sales/blob/main/img/Machine_Learning_Performance.PNG">

### Deploy Model to Production

-  Saving trained model
-  Building Rossmann Class
-  Building API tester

### Telegram Bot



###  Conclusion

Finally, it is possible to understand that although the model based on averages is simple, it is still coherent however not sensible enogh to comprehend store's oscilations, but if machine learn models were too expensive it would be a good choice to solve this problem, of course the data scientist and the CFO should consider its leak of precision in some points.

The XGBoost Model for the first cycle (CRISP-DM Methodology) presented a result within the acceptable range, although some stores were difficult to have the expected behavior presenting the MAPE (Mean Absolute Percentage Error) between 0.30 to 0.56, this first result it will be presented to the company, to inform the project status and what is already available as a solution.

###  Next Steps

Start a second cycle to analyze the problem, seeking different approaches, especially considering stores with behavior that is difficult to predict. In these stores the Data scientist ought gain plenty of experience.

Possible points to be addressed in the second cycle:

- Work with NA data differently

- Rescaling and Encoding of data with different methodologies

- Work with new features for forecasting

- Work with a more robust method to find the best Hyper parameters for the model

<br>

##  TOOLS

- Flask
- Scipy
- Pyenv
- Numpy
- Boruta
- Pandas
- Heroku
- XGBoost
- Sklearn
- Seaborn
- IPython
- Telegram
- Inflection
- Matplotlib
