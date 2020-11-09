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
<div class="imgcap">
<img src="/assets/bio/house_graph.jpg">
</div>

<img src="/assets/bio/house_graph.jpg" alt="house_graph" />

In this demonstration, we will be solving the problem of predicting house prices based on their various features. There are a plethora of factors which determine the price of a house, many of which are more significant than we might expect. By using a machine learning model to predict prices, we allow the most significant factors to be considered rather than solely relying on what we deem to be relevant. 

The data we will be using is the [Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf), which contains a wide number of features of the houses, ranging from their location to the height of their basements! 

The full code from this example is available on [GitHub](http://jse.amstat.org/v19n3/decock.pdf), along with a more [elegant solution](http://jse.amstat.org/v19n3/decock.pdf) using pipelines.

**Data Acquisition**

To get started, we simply need to load the data set into a Pandas dataframe. 

```
df = pd.read_csv("../data/train.csv")  
```


By using df.head() and df.info() we can quickly gauge the nature of the dataset. By exploring the data further, we can see that each of the 1460 data points represent unique houses.

**Exploration**

To start we’re going to look at the Sale Price (in $USD) for each property, as this is what we are predicting. 

<img src="/assets/bio/price_skew.png" alt="price_skew" style="width:100%"/>




We can see that the data is skewed to the left. This is not surprising as it reflects the distribution of wealth, resulting in a small number of expensive houses while the bulk remain at the lower end of the price spectrum. 

We must now explore which features are most relevant when determining the price. An easy way to do this is to look at the correlation of each variable with Sale Price. 

The correlations of the 4 chosen continuous features are as follows:

- Overall Quality (1-10, where 10 is excellent): Correlation = 0.791
- Ground Floor Area (square feet): Correlation = 0.709
- Garage Car Capacity: Correlation = 0.640
- Year Built = 0.523

In addition, 3 strongly correlated categorical features are:
- External Quality
- Heating Quality
- Kitchen Quality

These each fall into 5 categories assessing Quality. We can see the correlation between External Quality and Sale Price below:

<div class="imgcap">
<img src="/assets/bio/correlation.png">
</div>

It is important to consider deal with missing data but this isn’t always necessary. None of the features we have chosen have any missing data, so we can move on…

**Training and Test Set**

An important part of machine learning is to train the model only using TRAINING data. It’s best to set aside a subset of the data for testing as soon as possible. In this case we will use 80% of the set for training data and 20% of the test for testing data. Note that we shuffle the data before splitting. 
```
X = df['OverallQual', 'YearBuilt', 'ExterQual', 'HeatingQC', 'KitchenQual', 'GrLivArea', 'GarageCars']
Y=df['SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
```

**Data Preparation** 

Before we can jump into predicting house prices, we need to take a closer look at the data we’re using and how it should be represented. We’ve chosen features containing two types of data - categorical and continuous. These need to be dealt with differently. 

**One-Hot Encoding**

In order to represent categorical data in a format the model can understand, we use encoding. 

One method would be to represent each category as a number, for example the Quality of Build could be represented numerically, where 1=Ex, 2=Gd, 3=Fa and 4=Ta. This would however place more importance on data with greater numerical values. 

To avoid ording the categorical data, we use One-Hot encoding. This encodes each of the categories as columns of binary data as shown below. 

<div class="imgcap">
<img src="/assets/bio/one-hot.png">
</div>

It is important to use the same encoder for the test set as for the training set. This deal with any cases where a category is found in the test set which was not present in the training set. 

Below, `X_train_cat` and `X_test_cat` are the relivent categorical data for the training and test sets.
```
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(X_train_cat)
X_train_onehot = enc.transform(X_train_cat)
X_test_onehot = enc.transform(X_test_cat)
```
**Continuous Data**

The process of deciding **how to best represent** the continuous data is a bit more involved. 

Let’s take a look at the column `YearBuilt`. There are a couple of options of how we could represent this:

- We could choose to make it a categorical feature, having a category per year. This would avoid losing the information about the actual year each house was built, which may be significant. For example, Georgan houses may typically be worth a different amount to houses built in the Victorian Era. The creation of so many categories however is unlikely to be helpful in our model. 

- A preferable approach would be to calculate the number of years old each house is at the point of sale. Data giving the year of sale of each house is not available, however.

- The chosen approach is therefore to calculate the relative age of each house. This is done by comparing the year built of the newest house in the training data (2010) to each house. It is important to check that the test set is also compared to 2010, rather than to the newest house in the test set.   

```
most_recent = max(X_train['YearBuilt'])
X_train['YearBuilt'] = abs(X_train['YearBuilt']-most_recent)
X_test['YearBuilt'] = abs(X_test['YearBuilt']-most_recent)
```
 
Secondly, we must consider the **distribution** of each of these features. The features `YearBuilt` and `GrLivArea` are both skewed. Skewness can be reduced by taking the log or squareroot of each value. In this case sqrt best reduces the skew for `YearBuilt` and log for `GrLivArea`. To avoid taking the log of zero, we add 1 to all values. 

```
X_train['YearBuilt'] = np.sqrt(X_train['YearBuilt'])
X_test['YearBuilt'] = np.sqrt(X_test['YearBuilt'])

X_train['GrLivArea'] = np.log(X_train['GrLivArea']+1)
X_test['GrLivArea'] = np.log(X_test['GrLivArea']+1)
```

Taking the log of the values reduces the skew from 0.60 to 0.04:

<div class="imgcap">
<img src="/assets/bio/skew_yr.png" style="width:49%">
<img src="/assets/bio/unskew_yr.png" style="width:49%">
</div>

Incontrast to x, we can see in taking the sqareroot of the x values reduces the skew from x to x.

Lastly, we must consider **normalising** the continuous data. This involves setting the mean of the data to zero and the standard deviation to one.  

Standardising the data is beneficial when using linear regression as it speeds up the process and makes it more numerically robust. It is not necessary for Random Forrest Regression, but as we have not decided upon a model yet, we will standardise the data.  

It is important to standardise the test data using the same scaler as the train data:

```
X_train_cont = X_train[['OverallQual', 'YearBuilt','GrLivArea', 'GarageCars']]
scaler = StandardScaler()
scaler.fit(X_train_cont)
X_train_scaled = scaler.transform(X_train_cont)

X_test_cont = X_test[['OverallQual', 'YearBuilt','GrLivArea', 'GarageCars']]
X_test_scaled = scaler.transform(X_test_cont)
```

Finally, we must join the continuous and categorical data together again:

```
X_test_cleaned = np.concatenate((X_test_onehot,X_test_scaled), axis=1)
X_train_cleaned = np.concatenate((X_train_onehot,X_train_scaled), axis=1)
```

**Evaluation Metric**

We need to choose an evaluation metric. This is the metric by which we measure the success of a prediction. For example, a basic evaluation metric could be to look at the absolute error between a predicted and actual house price. 

In this case, we will choose the logarithmic root mean square error (lrmse). By using a logarithmic metric, the errors in predicting expensive houses and cheap houses will have an equal effect on the result.

**Train Model**

We will try two machine learning methods - Random Forrest Regression and Linear Regression. Using Scikit-learn, it’s extremely quick to implement these models. 

1. Random Forrest Regression
```
model = RandomForestRegressor()
model.fit(X_train_cleaned,y_train)
```
2. Linear Regression
```
model = RandomForestRegressor()
model.fit(X_train_cleaned,y_train)
```

**Make Predictions on the Test Set**

Now that our models have learnt the relationship between the features and prices in the test set, we can test them out on the test set. 
```
predictions = model.predict(X_test_cleaned)
sns.regplot(y_test,predictions)
```
The plot below show the difference between the predicted prices and the actual prices when using Random Forrest Regression:
<div class="imgcap">
<img src="/assets/bio/results.png">
</div>

By testing both models, we find that the best choice is Random Forrest Regression, which has an accuracy of the 81.0% and lrmse of 0.20, while Linear Regression gives an accuracy of 78.2% and lrmse of 0.26. 

**Evaluation Metric**

We need to choose an evaluation metric. This is the metric by which we measure the success of a prediction. For example, a basic evaluation metric could be to look at the absolute error between a predicted and actual house price. 

In this case, we will choose the logarithmic root mean square error (lrmse). By using a logarithmic metric, the errors in predicting expensive houses and cheap houses will have an equal effect on the result.

**Improving our model**

We now have a model which predicts house prices with x% accuracy. This isn’t bad but there’s scope for improvement. 

We can do this using hyperparameter tuning. We can think of this as adjusting the settings of the model to get the best performance. 

The best way to do this is essentially to try out different combinations of hyperparameters (settings) and see which work best. Thankfully, we can do this quickly using Scikit-learn.

We need to identify which hyperparameters are most important for our model. Once we’ve found these, we can create a grid of hyperparameter combinations.

Rather than go through every combination of hyperparameters, we can use RandomizedSearchCV to try a random choice of combinations from the grid.

We can see that our hyperparameter tuning has improved the accuracy the model by x%. This may not seem like much but depending on the application of the model, this could represent millions of pounds to a company. 

**Overfitting**

Random Forests are prone to overfitting. We can minimise this problem using K-Fold Cross-Validation when using RandomizedSearchCV. In this case, we do this by setting cv=3. 

**Visualisations**

Lastly, we can consider our findings from the model. It is worth looking at the importance of each variable in predicting price. Below, the features used are ordered from most to least important. 

Overall built quality is clearly a strong indicator of Sale Price. An unexpectedly significant feature perhaps is……

So do we really need to consider the rest of the features at all? From running our model just  
Using x and y, we can see the prediction accuracy drop from x to x. So, yes - though it may not seem like it, our 8 features are all significant.

**Conclusion**

I hope this showed how simple it is to implement a machine learning model from start to finish! To improve the model we could use more of the features and consider feature engineering. As we’ve found, 90% of the work is in data preparation and it really is worth considering how to best represent the data. 

<div class="imgcap">
<img src="/assets/bio/combustion.jpeg" style="width:57%">
<img src="/assets/bio/combustion2.png" style="width:41%">
<div class="thecap"><a href="https://ib.bioninja.com.au/higher-level/topic-8-metabolism-cell/untitled/energy-conversions.html">Left</a>: Chemically, as far as inputs and outputs alone are concerned, burning things with fire is identical to burning food for our energy needs. <a href="https://www.docsity.com/en/energy-conversion-fundamentals-of-biology-lecture-slides/241294/">Right</a>: the complete oxidation of C-C / C-H rich molecules powers not just our bodies but a lot of our technology.</div>
</div>
