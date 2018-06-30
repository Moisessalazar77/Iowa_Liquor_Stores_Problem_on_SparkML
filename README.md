# Iowa Liquor Stores Problem on SparkML with Scala.


# Introduction

The objective of this project is to advise a hypothetical owner of liquor stores chain where to open new stores and where the best opportunities in the state of Iowa liquor market are. The data is comprised of historical sales and location information. 

# Data Wrangling and Data Munging

The dataset do not had many missing values and is professionally redacted but, in order to submit this information into a machine learning alghoritm it needs to be converted and arranged in a numeric format. A peculiarty to spark is that needs to
to process data as vectors. It sees all the row of the datset as a vector with an associated label hence, the need to change the name of the target variable column to "label". The following snippet of the codes is an examplot of converting the field expressed as dollar to a numeric entry, firt the dollar sign must be remove and then the character strinbg coverted to a double precision number. 
```                                                                                                                                                                              val df1=df_rdx.withColumn("Sale (Dollars)", regexp_replace($"Sale (Dollars)","\\$+", " "))

  val df2=df1.withColumn("State Bottle Retail", regexp_replace($"State Bottle Retail","\\$+", " "))

  val df3=df2.withColumn("Sale (Dollars)", $"Sale (Dollars)".cast(sql.types.DoubleType))

  val df4=df3.withColumn("State Bottle Retail", $"State Bottle Retail".cast(sql.types.DoubleType))

  val df_preprocessed=df4.withColumnRenamed("Sale (Dollars)","label")
```
After this step the data set is ready for more complex techniques as feature selection and engineering as well as further preprocessing.

# SparkML library .

In contrast to R or Python, Scala uses the get/set paradigm to assigned parameters to an object. The SparkML library has to API one for RDD(Resilient Distributed Dataset) and for DATAFRAME, the newest one, which some expets considered to be the future of Spark. Although the SparkML by any standard, a capable machine learning library but, still need more development in certains area to be on pair with other packages as Python and R but, this is a challange due to Spark backbone on paraller computing and 
cluster data distribution schema but something that for certain it will be tackle in the near future. The following code snippet is the initialization of a linear regression object, and the assigment of certain hyperparameters.
```
val lr= new LinearRegression()
        .setFitIntercept(true)      
        .setStandardization(true)       
        .setTol(0.007)
        .setMaxIter(500)        
        .setFeaturesCol("features")
```

The fitness of the model is evaluated using the R square metric which is a measure of how much of the variance in the target(sales) can be explained by the model. In other words, if the sample change, the random variations will affect the predictions but, if the model is well fitted those random variations will be miniscule and the model still will be able to make acceptable predictions. Of course, all models are wrong, but some are useful!  

# Tuning a model with Scala

```
val lr= new LinearRegression()
        .setFitIntercept(true)      
        .setStandardization(true)       
        .setTol(0.007)
        .setMaxIter(500)        
        .setFeaturesCol("features")
```

# Conclusions

The Variables city and price alone can explain the 20% of the variance, making those features the more statistically significant.

The MAE (Median Absolute error) is close to 1.7 thousand dollars with is small in comparison with the mean of target.

As expected the big cities like Des Moines are at the top but the best opportunities are on the upcoming markets like Ankeny witch population has duplicated very recently.

# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

