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

val df_reduced = df.drop($"Invoice/Item Number")
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
                   

df.withColumn("Sale (Dollars)", regexp_replace($"Sale (Dollars)", "\\$+", " "))

df.select(df("Sales(Dollars)").cast(FloatType))

/*df.na.drop()*/


                   
                   
 df_reduced.show(5) 
 

 val le1 =new StringIndexer()
              .setInputCol("County")
              .setOutputCol("County_lb")
              
  val le2 =new StringIndexer()
              .setInputCol("Category")
              .setOutputCol("Category_lb")              
 
val le3 =new StringIndexer()
              .setInputCol("Item Description")
              .setOutputCol("Item Description_lb")
              
val enc1 = new OneHotEncoderEstimator()
              .setInputCols(Array("County_lb","Category_lb","Item Description_lb"))
              .setOutputCols(Array("CountyVec1", "CategoryVec2","Item_DescriptionVec3"))
              
             
                          
              
var assembler= new VectorAssembler()
                   .setInputCols(Array("Store Number",
                                        "County",
                                        "Category Name",
                                        "Vendor Number",
                                        "Item Description",
                                        "Bottle Volume (ml)",
                                        "State Bottle Retail", 
                                        "Bottles Sold"))
                     .setOutputCol("features")

var lr= new LinearRegression()
        .setFitIntercept(true)      
        .setStandardization(true)       
        .setTol(0.007)
        .setMaxIter(300)        
        .setFeaturesCol("features")
        
var pipeline = new Pipeline()
                   .setStages(Array())
  
val reg_pararm= Array(0.001,0.1,1,10)

val alpha = Array(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)

val loss_func =  Array("squaredError","huber")

val solvers = Array("l-bfgs","normal")
        
var paramGrid = new ParamGridBuilder()
                    .addGrid(lr.regParam,reg_pararm)
                    .addGrid(lr.elasticNetParam,alpha)
                    .addGrid(lr.loss,loss_func)
                    .addGrid(lr.solver,solvers)
                    .build()
  

var r2Evaluator = new RegressionEvaluator()
                      .setMetricName("r2")
  
var cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEstimatorParamMaps(paramGrid)
            .setEvaluator(r2Evaluator)

var Array(train, test) = df.randomSplit(weights=Array(.8, .2), seed=42)

var cvModel = cv.fit(train)

var predictions = cvModel.transform(test)

r2Evaluator.evaluate(predictions)

              

    
 
}

 
 
      


