package com.salazaraiassociates.spark


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{regexp_replace, regexp_extract}
import org.apache.spark.ml.feature.{StringIndexer,OneHotEncoderEstimator,VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object lr_model {
  
/** Our main function where the action happens */
  
 
Logger.getLogger("org").setLevel(Level.ERROR)

              
 val conf = new SparkConf()
            .setAppName("Iowa_liquor_Stores_lr_model")
            .setMaster("local[*]")
            
            
val sc= new SparkContext(conf)           

val sqlcontext = new SQLContext(sc)
           
             
 import sqlcontext.implicits._

  
val df = sqlcontext

     .read

     .format("csv")

     .option("header", "true")

     .option("inferSchema", "true")
 
     .option("delimiter", ",")

     .load("C:/iowa_liquor_store_problem/iowa_liquor_sales.csv")
      
df.show(7)
 
println(df.count())

df.columns

df.printSchema()

val df_rdx = df.drop($"Invoice/Item Number")
                   .drop($"Date")
                   .drop("Store Name")
                   .drop($"Address")
                   .drop($"City")
                   .drop($"Pack")
                   .drop($"State Bottle Cost")
                   .drop($"Zip Code")
                   .drop($"County Number")
                   .drop($"Store Location")
                   .drop($"Category")
                   .drop($"Vendor Name")
                   .drop($"Item Number")
                   .drop($"State Bottle Cost")
                   .drop($"Volume Sold (Liters)")
                   .drop($"Volume Sold (Gallons)")
                   

df_rdx.show(5)

  df_rdx.printSchema()

  val df1=df_rdx.withColumn("Sale (Dollars)", regexp_replace($"Sale (Dollars)","\\$+", " "))

  val df2=df1.withColumn("State Bottle Retail", regexp_replace($"State Bottle Retail","\\$+", " "))

  val df3=df2.withColumn("Sale (Dollars)", $"Sale (Dollars)".cast(sql.types.DoubleType))

  val df4=df3.withColumn("State Bottle Retail", $"State Bottle Retail".cast(sql.types.DoubleType))

  val df_preprocessed=df4.withColumnRenamed("Sale (Dollars)","label")

  df_preprocessed.printSchema()

  val le1 =new StringIndexer()
              .setInputCol("County")
              .setOutputCol("County_lb")
              .setHandleInvalid("keep")
              
              
  val le2 =new StringIndexer()
              .setInputCol("Category Name")
              .setOutputCol("Category_Name_lb")
              .setHandleInvalid("keep")
              
 
  val le3 =new StringIndexer()
              .setInputCol("Item Description")
              .setOutputCol("Item_Description_lb")
              .setHandleInvalid("keep")
              
                    
  val enc1 = new OneHotEncoderEstimator()
              .setInputCols(Array("County_lb","Category_Name_lb","Item_Description_lb"))
              .setOutputCols(Array("CountyVec1", "CategoryNameVec2","ItemDescriptionVec3"))
                    
              
              
  val assembler= new VectorAssembler()
                   .setInputCols(Array( "Store Number",
                                        "CountyVec1", 
                                        "CategoryNameVec2",
                                        "ItemDescriptionVec3",
                                        "Vendor Number",
                                        "Bottle Volume (ml)",
                                        "State Bottle Retail", 
                                        "Bottles Sold"))
                    .setOutputCol("features")


  val lr= new LinearRegression()
        .setFitIntercept(true)      
        .setStandardization(true)       
        .setTol(0.007)
        .setMaxIter(500)        
        .setFeaturesCol("features")
        
        
  val Costumed_Stages=Array(le1,le2,le3,enc1,assembler,lr)
        
  val pipeline = new Pipeline()
                   .setStages(Costumed_Stages)
  
  val reg_param= Array(0.001,0.1,1,10)

  val alpha = Array(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)

  val loss_func =  Array("squaredError")

  val solvers = Array("l-bfgs","normal")
        
  val paramGrid = new ParamGridBuilder()
                    .addGrid(lr.regParam,reg_param)
                    .addGrid(lr.elasticNetParam,alpha)
                    .addGrid(lr.loss,loss_func)
                    .addGrid(lr.solver,solvers)
                    .build()
  

  val r2Evaluator = new RegressionEvaluator()
                      .setMetricName("r2")
                      
  val maeEvaluator = new RegressionEvaluator()
                      .setMetricName("mae")
                      
  val mseEvaluator = new RegressionEvaluator()
                      .setMetricName("mse")
                 
  
  val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEstimatorParamMaps(paramGrid)
            .setEvaluator(r2Evaluator)

  val Array(train, test) = df_preprocessed.randomSplit(weights=Array(.8, .2), seed=42)

  val cvModel = cv.fit(train)

  val predictions = cvModel.transform(test)

  println(r2Evaluator.evaluate(predictions))

  println(maeEvaluator.evaluate(predictions))

  println(mseEvaluator.evaluate(predictions))
 

    
 
}

 
 
      


