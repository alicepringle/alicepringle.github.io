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
<div class="thecap">[Image source](https://www.propertyreporter.co.uk/property/rightmove-predict-2-house-price-rise-in-2020.html).</div>
</div>

In this demonstration, we will be solving the problem of predicting house prices based on their various features. There are a plethora of factors which determine the price of a house, many of which are more significant than we might expect. By using a machine learning model to predict prices, we allow the most significant factors to be considered rather than solely relying on what we deem to be relevant. 

The data we will be using is the [Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf), which contains a wide number of features of the houses, ranging from their location to the height of their basements! 

The full code from this example is available on [GitHub](http://jse.amstat.org/v19n3/decock.pdf), along with a more [elegant solution](http://jse.amstat.org/v19n3/decock.pdf) using pipelines.

**Data Acquisition**

To get started, we simply need to load the data set into a Pandas dataframe. 

<div class="imgcap">
<img src="/assets/bio/Screenshot 2020-11-07 at 16.26.22.png">
<div class="thecap">[Image source](https://www.propertyreporter.co.uk/property/rightmove-predict-2-house-price-rise-in-2020.html).</div>
</div>

```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
  all_data = pd.read_csv("../data/train.csv")  
  df = pd.DataFrame(data=all_data)
}
```


By using df.head and df.info() we can quickly gauge the nature of the dataset. By exploring the data further, we can see that each of the 1460 data points represent unique houses.

**Exploration**

To start we’re going to look at the Sale Price (in $USD) for each property, as this is what we are predicting. 

<div class="imgcap">
<img src="/assets/bio/Screenshot 2020-11-07 at 16.26.22.png">
<div class="thecap">Distribution Graph.</div>
</div>

We can see that the data is skewed to the left. This is not surprising as it reflects the distribution of wealth, resulting in a small number of expensive houses while the bulk remain at the lower end of the price spectrum. 

We must now explore which features are most relevant when determining the price. An easy way to do this is to look at the correlation of each variable with Sale Price. 

The first row of the plot above shows the 8 most strongly correlated features. These features are as follows:

- year built
- year built
- year built

It is important to consider deal with missing data but this isn’t always necessary. None of the features we have chosen have any missing data, so we can move on…


**Training and Test Set**

An important part of machine learning is to train the model only using TRAINING data. It’s best to set aside a subset of the data for testing as soon as possible. In this case we will use x% of the set for training data and y% of the test for testing data. 


**Data Preparation** 

Before we can jump into predicting house prices, we need to take a closer look at the data we’re using and how it should be represented. We’ve chosen features containing two types of data - categorical and continuous. These need to be dealt with differently. 

**One-Hot Encoding**

In order to represent categorical data in a format the model can understand, we use encoding. 

One method would be to represent each category as a number, for example the Quality of Build could be represented numerically, where 1=Ex, 2=Gd, 3=Fa and 4=Ta. This would however place more importance on data with greater numerical values. 

To avoid ording the categorical data, we use One-Hot encoding. This encodes each of the categories as columns of binary data as shown below. 

It is important to use the same encoder for the test set as for the training set. This deal with any cases where a category is found in the test set which was not present in the training set.

**Continuous Data**

The process of deciding how to best represent the continuous data is a bit more involved. 

Let’s take a look at the column YearBuilt. There are a couple of options of how we could represent this. 

We could choose to make it a categorical feature, having a category per year. This would avoid losing the information about the actual year each house was built, which may be significant. For example, Georgan houses may typically be worth a different amount to houses built in the Victorian Era. The creation of so many categories however is unlikely to be helpful in our model. 

The chosen approach is therefore to calculate the relative age of each house. This is done by comparing the year built of the newest house in the training data (2010) to each house. It is important to check that the test set is also compared to 2010, rather than to the newest house in the test set.   

A preferable approach would be to calculate the number of years old each house is at the point of sale. Data giving the year of sale of each house is not available, however.
 
Secondly, we must consider the distribution of each of these features. We can seen that x,x,x,x are each skewed, to varying degrees. This can be adjusted by taking the log or squareroot of each value.

We can see that taking the log of the values reduces the skew from x to x. 

Incontrast to x, we can see in taking the sqareroot of the x values reduces the skew from x to x.

Lastly, we must consider normalising the continuous data. This involves setting the mean of the data to zero and the standard deviation to one.  

Standardising the data is beneficial when using linear regression as it speeds up the process and makes it more numerically robust. It is not necessary for Random Forrest Regression, but as we have not decided upon a model yet, we will standardise the data.  

It is important to standardise the test data using the same scaler as the train data. This is done below.

**Evaluation Metric**

The last thing to do before training our model is to choose an evaluation metric. This is the metric by which we measure the success of a prediction. For example, a basic eva,uation metric could be to look at the absolute error between a predicted and actual house price. 

In this case, we will choose the logarithmic root mean square error (lrmse). By using a logarithmic metric, the errors in predicting expensive houses and cheap houses will have an equal effect on the result.

**Train Model**

We will try two machine learning methods - Random Forrest Regression and Linear Regression. Using Scikit-learn, it’s extremely quick to implement these models. 

1. Random Forrest Regression
2. Linear Regression

**Make Predictions on the Test Set**

Now that our models have learnt the relationship between the features and prices in the test set, we can test them out on the test set. 

To do this we must make predictions on the test set and use the evaluation metric to compare them to the correct house prices.  

By doing this we can see that the best choice is Random Forrest Regression, which has an accuracy of the x%, while Linear Regression gives an accuracy of x%. 

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

> We've now established in some detail that fat is your body's primary battery pack and we'd like to breathe it out. Let's turn to the details of the accounting.

 St. Jeor for men is *10 x weight (kg) + 6.25 x height (cm) - 5 x age (y) + 5*). Anyone who's been at the gym and ran on a treadmill will know just how much of a free win this is. I start panting and sweating uncomfortably just after a small few 


<pre style="font-size:10px">
2019-09-23: Morning weight 180.5. Ate 1700, expended 2710 (Δkcal 1010, Δw 0.29). Tomorrow should weight 180.2
2019-10-01: Morning weight 179.4. Ate 2000, expended 2637 (Δkcal 637, Δw 0.18). Tomorrow should weight 179.2
2019-10-02: Morning weight 179.5. Ate 1920, expended 2552 (Δkcal 632, Δw 0.18). Tomorrow should weight 179.3
</pre>
