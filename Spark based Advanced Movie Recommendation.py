#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install pyspark


# In[ ]:


# from zipfile import ZipFile
# import os

# !wget https://files.grouplens.org/datasets/movielens/ml-25m.zip

# # Unzip the downloaded file
# with ZipFile('ml-25m.zip', 'r') as zip_ref:
#     zip_ref.extractall()


# os.listdir('ml-25m')


# In[1]:


from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, sum, expr, count, udf, collect_list,  explode, split, regexp_extract
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
from pyspark.sql.types import  FloatType, IntegerType


# In[2]:


# spark = SparkSession.builder.appName("MovieRecommender")\
#         .config("spark.driver.memory", "120g")\
#         .config("spark.executor.memory", "120g")\
#         .config("spark.local.dir", "/home/robotixx/Documents/Aniket/assignment 4")\
#         .getOrCreate()
spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# In[3]:


movies = spark.read.csv("ml-25m/movies.csv", header=True, inferSchema=True)
ratings = spark.read.csv("ml-25m/ratings.csv", header=True, inferSchema=True)
ratings=ratings.drop('timestamp')

movies.show(5)
ratings.show(5)


# In[4]:


# minRatingsPerUser = 10  # Threshold for users
# minRatingsPerMovie = 10  # Threshold for movies

# userCounts = ratings.groupBy("userId").agg(count("rating").alias("num_ratings"))
# activeUsers = userCounts.filter(col("num_ratings") >= minRatingsPerUser).select("userId")

# movieCounts = ratings.groupBy("movieId").agg(count("rating").alias("num_ratings"))
# popularMovies = movieCounts.filter(col("num_ratings") >= minRatingsPerMovie).select("movieId")

# filteredRatings = ratings.join(activeUsers, on="userId").join(popularMovies, on="movieId")

# sampledFilteredRatings = filteredRatings.sample(withReplacement=False, fraction=0.25, seed=42)
# # sampledFilteredRatings.count()
# sampledFilteredRatings = sampledFilteredRatings.repartition("userId")


# In[5]:


trainData, testData = ratings.randomSplit([0.8, 0.2], seed=123)


# In[6]:


# Configure ALS model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy ='drop'
)

# Train the ALS model
model = als.fit(trainData)


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
predictions = model.transform(testData)
predictions.show(10)

print("\n ************************** \n")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# In[7]:


# Configure ALS model
tuneALS = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy ='drop'
)

param_grid = ParamGridBuilder() \
    .addGrid(tuneALS.rank, [10, 15, 20]) \
    .addGrid(tuneALS.maxIter, [5, 10, 15]) \
    .addGrid(tuneALS.regParam, [0.1, 0.01, 0.001]) \
    .build()


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")


crossValidator = CrossValidator(
    estimator=tuneALS,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
)

trainData = trainData.repartition("userId")
testData = testData.repartition("userId")


cvModel = crossValidator.fit(trainData)
bestModel = cvModel.bestModel

bestRank = bestModel.rank
bestMaxIter = bestModel._java_obj.parent().getMaxIter()
bestRegParam = bestModel._java_obj.parent().getRegParam()

tunePredictions = bestModel.transform(testData)
tunePredictions.show(5)

print("\n ************************** \n")
print("\nALS Best Accuracy and Parameters\n")
print(f"ALS - Best rank: {bestRank}")
print(f"ALS - Best maxIter: {bestMaxIter}")
print(f"ALS - Best regParam: {bestRegParam}")


rmse = evaluator.evaluate(tunePredictions)
mse = evaluator.evaluate(tunePredictions, {evaluator.metricName: "mse"})


print("\n ************************** \n")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
count_to_take = 1000
ratings = ratings.limit(count_to_take)

def calculate_map(predictions, k):
    
    windowSpec = Window.partitionBy("userId").orderBy(col("prediction").desc())
    rankedPredictions = predictions.withColumn("rank", F.rank().over(windowSpec))
    topKpredictions = rankedPredictions.filter(col("rank") <= k)
    topKpredictions.createOrReplaceTempView("top_k_predictions")
    precisionK = spark.sql(f"""
        SELECT userId, 
               SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) / {k} as precisionK
        FROM top_k_predictions
        GROUP BY userId
    """)

    
    avgprecisionK = precisionK.agg(F.mean("precisionK").alias("map")).collect()[0]["map"]
    
    return avgprecisionK


mapScore = calculate_map(tunePredictions, 10)
print(f"Mean Average Precision for top-10 recommendations: {mapScore:.2f}")


# In[8]:


itemMatrix = ratings.groupBy("movieId").agg(F.collect_list("rating").alias("ratings"))
itemPairs = itemMatrix.alias("item1").join(itemMatrix.alias("item2"), F.col("item1.movieId") < F.col("item2.movieId"))


def cosine_similarity(ratings1, ratings2):
    dotProduct = F.expr('aggregate(transform(ratings1, (x, i) -> IFNULL(x, 0.0) * IFNULL(ratings2[i], 0.0)), 0D, (acc, x) -> acc + x)')
    norm1 = F.expr('sqrt(aggregate(transform(ratings1, x -> IFNULL(x, 0.0) * IFNULL(x, 0.0)), 0D, (acc, x) -> acc + x))')
    norm2 = F.expr('sqrt(aggregate(transform(ratings2, x -> IFNULL(x, 0.0) * IFNULL(x, 0.0)), 0D, (acc, x) -> acc + x))')
    return dotProduct / (norm1 * norm2)

explodedItemPairs = itemPairs.select(
    F.col("item1.movieId").alias("item1_movieId"),
    F.col("item2.movieId").alias("item2_movieId"),
    F.col("item1.ratings").alias("ratings1"),
    F.col("item2.ratings").alias("ratings2")
)

# Calculate cosine similarity while handling null values
itemSimilarities = explodedItemPairs.withColumn(
    "similarity",
    cosine_similarity(F.col("ratings1"), F.col("ratings2"))
).select("item1_movieId", "item2_movieId", "similarity")

similarityRatings = ratings.join(itemSimilarities, ratings.movieId == itemSimilarities.item1_movieId)

predictions = predictions.withColumnRenamed('rating', 'als_rating')
finalDataset = similarityRatings.join(predictions, on=["userId", "movieId"])


# Define hybrid prediction function using Spark DataFrame operations
def hybrid_predict(als_pred, similarity, rating, alpha=0.5):
    return (alpha * als_pred + (1 - alpha) * similarity * rating)

finalDataset = finalDataset.withColumn("hybrid_prediction", hybrid_predict(F.col("als_rating"), F.col("similarity"), F.col("rating")))


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="hybrid_prediction")
hybridRMSE = evaluator.evaluate(finalDataset)
print(f"Hybrid Model RMSE: {hybridRMSE:.2f}")


# In[9]:


# Split the genres string into a list of genres
movies = movies.withColumn("genre", explode(split("genres", "\|")))

# One-hot encoding of genres
indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
movies = indexer.fit(movies).transform(movies)

encoder = OneHotEncoder(inputCols=["genreIndex"], outputCols=["genreVec"])
movies = encoder.fit(movies).transform(movies)

movies = ratings.join(movies, on=["movieId"])
movies = movies.withColumn("year", regexp_extract("title", "\((\d+)\)", 1))


movies = movies.drop("rating","userId")
movieDataset = finalDataset.join(movies, on=["movieId"])
movieDataset = movieDataset.withColumn("year", col("year").cast(IntegerType()))


movieData = movieDataset.select("movieId", "userId", "year", "genreVec", "rating")
assembler = VectorAssembler(inputCols=["movieId", "userId", "year", "genreVec"], outputCol="features")
finalMovieDataset = assembler.transform(movieData)


# In[10]:


finalTrainData, finalTestdata = finalMovieDataset.randomSplit([0.8, 0.2], seed=123)


# In[11]:


#Decision Tree
dt = DecisionTreeRegressor(featuresCol="features", labelCol="rating", maxDepth=10, minInstancesPerNode=10, minInfoGain=0.01, maxBins=35)
dtModel = dt.fit(finalTrainData)
dtPredictions = dtModel.transform(finalTestdata)


enhancedDTDataset = finalDataset.join(dtPredictions.select("userId", "movieId", "prediction")\
                            .withColumnRenamed("prediction", "dt_prediction"), on=["userId", "movieId"])


def enhancedDTHybridPredict(alsPred, itemCFpred, dtPred, weightALS=1.0, weightItemCF=9.0, weightDT=8.0):
    totalWeight = weightALS + weightItemCF + weightDT
    return (weightALS * alsPred + weightItemCF * itemCFpred + weightDT * dtPred) / totalWeight

enhancedDTHybridPredictUDF = udf(enhancedDTHybridPredict, FloatType())

enhancedDTDataset = enhancedDTDataset.withColumn("enhanced_DT_hybrid_prediction", \
                        enhancedDTHybridPredictUDF(col("prediction"), col("hybrid_prediction"),\
                                                     col("dt_prediction")))


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="enhanced_DT_hybrid_prediction")
enhancedHybridDT_RMSE = evaluator.evaluate(enhancedDTDataset)
print(f"Enhanced Hybrid Decision Tree Model RMSE: {enhancedHybridDT_RMSE:.2f}")


# In[19]:


#GBTRegressor
gbt = GBTRegressor(featuresCol="features", labelCol="rating", maxIter=100, maxDepth=10, stepSize=0.5,
                   minInstancesPerNode=1, minInfoGain=0.01, lossType="squared")

gbtModel = gbt.fit(finalTrainData)
gbtPredictions = gbtModel.transform(finalTestdata)


enhancedGBTDataset = finalDataset.join(gbtPredictions.select("userId", "movieId", "prediction")\
                            .withColumnRenamed("prediction", "gbt_prediction"), on=["userId", "movieId"])


def enhancedGBTHybridPredict(alsPred, itemCFpred, gbtPred, weightALS=1.0, weightItemCF=9.0, weightGBT=7.0):
    totalWeight = weightALS + weightItemCF + weightGBT
    return (weightALS * alsPred + weightItemCF * itemCFpred + weightGBT * gbtPred) / totalWeight

enhancedGBTHybridPredictUDF = udf(enhancedGBTHybridPredict, FloatType())

enhancedGBTDataset = enhancedGBTDataset.withColumn("enhanced_GBT_hybrid_prediction", \
                        enhancedGBTHybridPredictUDF(col("prediction"), col("hybrid_prediction"),\
                                                     col("gbt_prediction")))


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="enhanced_GBT_hybrid_prediction")
enhancedHybridGBT_RMSE = evaluator.evaluate(enhancedGBTDataset)
print(f"Enhanced Hybrid GBT Model RMSE: {enhancedHybridGBT_RMSE:.2f}")


# In[13]:


#Random Forest
rf = RandomForestRegressor(featuresCol="features", labelCol="rating", numTrees=50, maxDepth=10)
rfModel = rf.fit(finalTrainData)
rfPredictions = rfModel.transform(finalTestdata)


# Join the random forest output predictions with the finalDataset
enhancedFinalMovieDataset = finalDataset.join(rfPredictions.select("userId", "movieId", "prediction")\
                                 .withColumnRenamed("prediction", "rf_prediction"), on=["userId", "movieId"])


# hybrid prediction function includeing the Random Forest predictions
def enhanced_hybrid_predict(alsPred, itemCFpred, rfPred, weightALS=1.0, weightItemCF=8.0, weightRF=8.0):
    totalWeight = weightALS + weightItemCF + weightRF
    return (weightALS * alsPred + weightItemCF * itemCFpred + weightRF * rfPred) / totalWeight

enhancedHybridPredictUDF = udf(enhanced_hybrid_predict, FloatType())

enhancedFinalMovieDataset = enhancedFinalMovieDataset.withColumn("enhanced_hybrid_prediction", \
                            enhancedHybridPredictUDF(col("prediction"), col("hybrid_prediction"),\
                                                     col("rf_prediction")))



evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="enhanced_hybrid_prediction")
enhanced_hybrid_rmse = evaluator.evaluate(enhancedFinalMovieDataset)
print(f"Enhanced Hybrid Model RMSE: {enhanced_hybrid_rmse:.2f}")


# In[20]:

#Grpah Plotting

# import matplotlib.pyplot as plt

# als = rmse
# hybridAlsItem = hybridRMSE 
# enhancedDT = enhancedHybridDT_RMSE
# enhancedGBT = enhancedHybridGBT_RMSE
# enhancedRF = enhanced_hybrid_rmse


# models = ['ALS', 'Hybrid Model (ALS+Item-Item CF)', 'Enhanced Hybrid Model(Random Forest)', 'Enhanced Hybrid Model (Decision Tree)', 'Enhanced Hybrid Model (GBT Regressor)']
# rmse_scores = [als, hybridAlsItem, enhancedDT, enhancedGBT, enhancedRF]

# plt.figure(figsize=(10, 6))
# plt.plot(models, rmse_scores, marker='o', color='red', linestyle='-', linewidth=2)


# plt.title('Comparison of Model RMSE Scores')
# plt.xlabel('Models')
# plt.ylabel('RMSE Score')
# plt.xticks(rotation=90)


# plt.grid(True)
# plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()


# In[ ]:




