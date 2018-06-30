# Iowa Liquor Stores Problem on SparkML with Scala."


# Introduction

The objective of this project is to advise a hypothetical owner of liquor stores chain where to open new stores and where the best opportunities in the state of Iowa liquor market are. The data is comprised of historical sales and location information. 

# Data Wrangling and Data Munging
The dataset has features like zip code, address, county and sales (the target variable). The information is self-explanatory and the gross majority features had just a few missing values. The first lines of code result are to convert the feature data type to a data type that can be input into a model. For instance, something like the following:
```
dataset_with_target['Sale (Dollars)'] = dataset_with_target['Sale (Dollars)'].replace({'\$':''}, regex=True)
```

At the core, preprocessing this dataset was mainly transforming the initial data types to a numeric form or converting measuring unit to a single unit to unified different presentations of the same physical variable, volume in this case.


# Multivariate linear regression Model.

This statistical model has a proven record in making prediction on sales hence the reason it was chosen for the project. In layman terms the model used the correlation between the features and the target in order to make a prediction such that it minimized the MSE(mean square error).

The fitness of the model is evaluated using the R square metric which is a measure of how much of the variance in the target(sales) can be explained by the model. In other words, if the sample change, the random variations will affect the predictions but, if the model is well fitted those random variations will be miniscule and the model still will be able to make acceptable predictions. Of course, all models are wrong, but some are useful!  

# Conclusions

The Variables city and price alone can explain the 20% of the variance, making those features the more statistically significant.

The MAE (Median Absolute error) is close to 1.7 thousand dollars with is small in comparison with the mean of target.

As expected the big cities like Des Moines are at the top but the best opportunities are on the upcoming markets like Ankeny witch population has duplicated very recently.

# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

