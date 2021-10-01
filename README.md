![Rossmann Store Sales](https://user-images.githubusercontent.com/91201232/134559863-1b84aae6-b027-4fd0-9bf6-8be122d0ec83.png)
<br>
<br>

## CONTENTS
- [Business](#busuness)
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

 - [x] Data Description
 - [x] Feature Engineering
 - [x] Variable Filtering
 - [x] Exploratory Data Analysis (EDA)
 - [x] Data Preparation
 - [x] Feature Selection
 - [x] Machine Learning Modelling
 - [x] Hyperparameter Fine Tuning
 - [x] Interpretation of the Error
 - [x] Deploy Model
<br>
 
## PROGRESS

### Data Description
- Understanding the business
- CRISP-DS Steps
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
- Multivariate analysis on numerical and categorical variables using heatmap for numerical variables and Cramér V calculation for categorical variables.

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

### Interpretation of the Error


### Deploy Model

##  TOOLS

- Flask
- Scipy
- Numpy
- Pyenv
- Boruta
- Pandas
- Heroku
- XGBoost
- Sklearn
- Seaborn
- IPython
- Inflection
- Matplotlib
