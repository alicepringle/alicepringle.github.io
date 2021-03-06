---
layout: post
comments: false
title:  "Predicting House Prices"
excerpt: "An End-to-End Machine Learning Example."
date:   2020-11-07 10:00:00
mathjax: false
---
<!-- <div class="imgcap">
<img src="/assets/bio/house_coin.jpg">
<div class="thecap">[Image source](https://www.propertyreporter.co.uk/property/rightmove-predict-2-house-price-rise-in-2020.html).</div>
</div> -->

<img src="/assets/bio/house_coin.jpg" alt="house_coin" style="width:95%"/>

The general framework when solving a problem using machine learning is pretty consistent. No matter the complexity of the problem, we need to consider a number of core principles, from how we treat training/ test data, to the representation of different features. 

In this post we will run through a machine learning example from start to finish. The aim is to show an comprehensive process which can be applied to a number of problems.   

**Problem Introduction**

We will be tackling the problem of predicting house prices based on their features. There are a plethora of factors which determine the price of a house, many of which are more significant than we might expect. By using machine learning to predict prices, we can allow the most significant factors to be considered rather than relying on what we deem to be relevant.

The data we will be using is the [Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf), which contains a wide number of features of the houses, ranging from their location to the height of their basements! 

The full code from this example is available on [GitHub](https://github.com/alicepringle/super-learning/blob/main/HousePrices.ipynb).

**Data Acquisition**

To get started, we simply need to load the data set into a Pandas dataframe. 

```
df = pd.read_csv("../data/train.csv")  
```

By using [df.head()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html) and [df.info()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html) we can quickly gauge the nature of the dataset. It's worth checking that each of the 1460 data points represent unique houses. Luckily for us, they do.

**Exploration**

To start off it makes to look at the variable we're predicting, in this case the Sale Price ($USD).

<img src="/assets/bio/price_skew.png" alt="price_skew" style="width:100%"/>

We can see that the data is positively skewed. This isn't surprising as we'd expect house prices to reflect the distribution of wealth. It makes sense, therefore, that there are a small number of expensive houses while the bulk remain at the lower end of the price spectrum. 

With 79 features, it would be helpful to explore **which features are most relevant** when determining the price. An easy way to do this is to look at the correlation of each variable with `SalePrice`. 

The correlation of the 4 chosen continuous features is as follows:

- Overall Quality (1-10, where 10 is excellent): Correlation = 0.791
- Ground Floor Area (square feet): Correlation = 0.709
- Garage Car Capacity: Correlation = 0.640
- Year Built = 0.523

In addition, 3 strongly correlated categorical features are:
- External Quality 
- Heating Quality 
- Kitchen Quality 

These each fall into 5 categories assessing Quality: Ex, Gd, Fa, Ta. We can see the correlation between External Quality and Sale Price below:

<img src="/assets/bio/correlation.png" alt="correlation" style="width:100%"/>

Before we move on we need to consider missing data. In this case, none of the features we have chosen have any missing data so we can move on… 

**Training and Test Set**

To estimate our model's performance on unseen data, we randomly reserve 20% of the houses for evaluation. The remaining 80% is used to train the model. It’s best to set aside a subset of the data for testing as soon as possible.
```
X = df[['OverallQual', 'YearBuilt', 'ExterQual', 'HeatingQC', 'KitchenQual', 'GrLivArea', 'GarageCars']]
Y=df['SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
```
**Data Preparation** 

Before we can jump into predicting house prices, we need to take a closer look at the data we’re using and how it should be represented. We’ve chosen features containing two types of data - categorical and continuous. These need to be dealt with differently. 

***Categorical Data: One-Hot Encoding***

In order to represent categorical data in a format the model can understand, we use encoding. In order to avoid ordering the categorical data, we use One-Hot encoding. This encodes each of the categories as columns of binary data, as shown below. 

<img src="/assets/bio/one-hot.png" alt="one-hot" style="width:100%"/>

It's important to use the same encoder for the test set as for the training set. This allows us to deal with any cases where a category is found in the test set which was not present in the training set. 

```
X_train_cat=X_train[['ExterQual', 'HeatingQC', 'KitchenQual']]
X_test_cat=X_test[['ExterQual', 'HeatingQC', 'KitchenQual']]

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(X_train_cat)

X_train_onehot = enc.transform(X_train_cat)
X_test_onehot = enc.transform(X_test_cat) 
```
***Continuous Data***

The process of deciding **how to best represent** the continuous data is a bit more involved. 

Let’s take a look at the column `YearBuilt`. There are a couple of options of how we could represent this:

- We could choose to make it a categorical feature, having a category per year. This would avoid losing the information about the actual year each house was built, which may be significant. For example, Georgian houses may typically be worth a different amount to houses built in the Victorian Era. The creation of so many categories, however, is unlikely to be helpful in our model. 

- A preferable approach would be to calculate the number of years old each house is at the point of sale. Unfortunately, we don't have data about the year of sale of each house so this isn't possible.

- The best option is therefore to calculate the relative age of each house. This is done by comparing the year built of the newest house in the training data (2010) to each house. It's important to check that the test set is also compared to 2010, rather than to the newest house in the test set.   

```
most_recent = max(X_train['YearBuilt'])
X_train['YearBuilt'] = abs(X_train['YearBuilt']-most_recent)
X_test['YearBuilt'] = abs(X_test['YearBuilt']-most_recent)
```
 
Secondly, we need to look at the **distribution** of each of these features. The features `YearBuilt` and `GrLivArea` are both skewed. Skewness can be reduced by taking the log or square root of each value. In this case, sqrt best reduces the skew for `YearBuilt` and log for `GrLivArea`. To avoid taking the log of zero, we add 1 to all values. 

```
X_train['YearBuilt'] = np.sqrt(X_train['YearBuilt'])
X_test['YearBuilt'] = np.sqrt(X_test['YearBuilt'])

X_train['GrLivArea'] = np.log(X_train['GrLivArea']+1)
X_test['GrLivArea'] = np.log(X_test['GrLivArea']+1)
```

Taking the log of the `YearBuilt` values reduces the skew from 0.60 to 0.04:

<div class="imgcap">
<img src="/assets/bio/skew_yr.png" style="width:49%">
<img src="/assets/bio/unskew_yr.png" style="width:49%">
</div>

Lastly, it's helpful to **standardise** the continuous data. This involves setting the mean of the data to zero and the standard deviation to one.  

This is beneficial when using linear regression as it reduces the number of iterations gradient descent takes to converge and makes it more numerically robust. It's not necessary for Random Forrest Regression, but as we haven't decided upon a model yet, let's standardise the data.  

When doing this, we need to ensure we use the same for the test data as we did for the train data:

```
X_train_cont = X_train[['OverallQual', 'YearBuilt','GrLivArea', 'GarageCars']]
X_test_cont = X_test[['OverallQual', 'YearBuilt','GrLivArea', 'GarageCars']]

scaler = StandardScaler()
scaler.fit(X_train_cont)
X_train_scaled = scaler.transform(X_train_cont)
X_test_scaled = scaler.transform(X_test_cont)
```

Finally, we can join the continuous and categorical data together again:

```
X_test_cleaned = np.concatenate((X_test_onehot,X_test_scaled), axis=1)
X_train_cleaned = np.concatenate((X_train_onehot,X_train_scaled), axis=1)
```

**Evaluation Metric**

Next, let's choose an evaluation metric. This is the metric by which we measure the success of a prediction. For example, a basic evaluation metric could be the absolute error between the predicted and actual house prices. 

In this case, we will choose the Logarithmic Root Mean Square Error (lrmse). By using a logarithmic metric, the errors in predicting expensive houses and cheap houses will have an equal effect on the result.

**Train Model**

We will try two machine learning methods - **Random Forrest Regression** and **Linear Regression**. Using Scikit-learn, it’s really quick to implement models, so we can try different ones out. 

1. Random Forrest Regression
```
model = RandomForestRegressor()
model.fit(X_train_cleaned,Y_train)
```
2. Linear Regression
```
model = LinearRegression()
model.fit(X_train_cleaned,Y_train)
```

**Make Predictions on the Test Set**

Now that our models have learnt the relationship between the features and prices in the test set, we can try them out on the test set. 
```
predictions = model.predict(X_test_cleaned)
sns.regplot(Y_test,predictions)
```
The plot below shows the difference between the predicted prices and the actual prices when using Random Forrest Regression:
<p align="center">
<img src="/assets/bio/results.png" alt="results" style="width:50%"/>
</p>
By testing both models, we find that the best choice is **Random Forrest Regression**, which had an accuracy of the 85.6% and lrmse of 0.15, while Linear Regression had an accuracy of 78.2% and lrmse of 0.26. 

**Hyperparameter tuning**

We now have a model which predicts house prices with 86% accuracy. This isn’t bad but there’s scope for improvement. 

We can do this using **hyperparameter tuning**. We can think of this as adjusting the settings of the model to get the best performance. 

The best way to do this is essentially to try out different combinations of hyperparameters (settings) and see which work best. Thankfully, we can do this quickly using Scikit-learn.

 We need to look up which hyperparameters are most important for our model from the API. Once we know these, we can create a grid of hyperparameter combinations. You can see how to create this grid on [GitHub](https://github.com/alicepringle/super-learning/blob/main/HousePrices.ipynb).

```
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
 ```

Rather than go through every combination of hyperparameters, we can use [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to try a random choice of combinations from the grid. This chooses the best combination according to our evaluation metric (lrmse).
```
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=scorer)
model_random.fit(X_train_cleaned,y_train)
```

Our hyperparameter tuning has reduced the lrmse by 3.26%. This may not seem like much but depending on the application of the model, this could represent millions of pounds to a company. You may notice that the accuracy of the model was marginally reduced but this model is preferable according to our evaluation metric.

**Cross-Validation**

[K-Fold Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) allows us to test on every house in our dataset, improving our estimate of how the model will perform on unseen data. For example, with 5-fold cross-validation, the model is trained on 4/5ths of the data and tested on the remaining 5th. This process is repeated 5 times until all the data has been used for evaluation. We can do this while using RandomizedSearchCV. In this case, we use 3 folds by setting cv=3. 

<p align="center">
<img src="/assets/bio/grid_search_cross_validation.png" alt="grid_search_cross_validation" style="width:60%"/>
</p>

**Conclusion**

So, we now have a model predicting house prices to 84% accuracy, with a lrmse of 0.149. Not a bad start. There a lots of things we can do to improve this model. The obvious next step would be to use more features - we have 79 to choose from! For now, I hope this has shown how simple it is to implement a machine learning model from start to finish.   