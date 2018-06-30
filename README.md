# Iowa Liquor Stores Problem on SparkML with Scala.


# Introduction

The objective of this project is to advise a hypothetical owner of liquor stores chain where to open new stores and where the best opportunities in the state of Iowa liquor market are. The data is comprised of historical sales and location information. 

# Data Wrangling and Data Munging

The dataset do not had many missing values and is professionally redacted but, in order to submit this information into a machine learning alghoritm it needs to be converted and arranged in a numeric format.
```                                                                                                                                                                              val df1=df_rdx.withColumn("Sale (Dollars)", regexp_replace($"Sale (Dollars)","\\$+", " "))

  val df2=df1.withColumn("State Bottle Retail", regexp_replace($"State Bottle Retail","\\$+", " "))

  val df3=df2.withColumn("Sale (Dollars)", $"Sale (Dollars)".cast(sql.types.DoubleType))

  val df4=df3.withColumn("State Bottle Retail", $"State Bottle Retail".cast(sql.types.DoubleType))

  val df_preprocessed=df4.withColumnRenamed("Sale (Dollars)","label")
```

At the core, preprocessing this dataset was mainly transforming the initial data types to a numeric form or converting measuring unit to a single unit to unified different presentations of the same physical variable, volume in this case.


# SparkML library .

This statistical model has a proven record in making prediction on sales hence the reason it was chosen for the project. In layman terms the model used the correlation between the features and the target in order to make a prediction such that it minimized the MSE(mean square error).

The fitness of the model is evaluated using the R square metric which is a measure of how much of the variance in the target(sales) can be explained by the model. In other words, if the sample change, the random variations will affect the predictions but, if the model is well fitted those random variations will be miniscule and the model still will be able to make acceptable predictions. Of course, all models are wrong, but some are useful!  

# Conclusions

The Variables city and price alone can explain the 20% of the variance, making those features the more statistically significant.

The MAE (Median Absolute error) is close to 1.7 thousand dollars with is small in comparison with the mean of target.

As expected the big cities like Des Moines are at the top but the best opportunities are on the upcoming markets like Ankeny witch population has duplicated very recently.

# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

