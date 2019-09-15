# 0. Data Exploration and Cleaning
df_clean=spark.read.csv("/FileStore/tables/animals_comments.csv", inferSchema=True, header=True)
df_clean.show(3)
df_clean.count() 
df_clean = df_clean.na.drop(subset=["comment"])
df_clean.count()
# Explore the data
df_clean.show()
# Label the data
# find user with preference of dog and cat
from pyspark.sql.functions import when
from pyspark.sql.functions import col

# you can user your ways to extract the label

df_clean = df_clean.withColumn("label", \
                           (when(col("comment").like("%my dog%"), 1) \
                           .when(col("comment").like("%I have a dog%"), 1) \
                           .when(col("comment").like("%my cat%"), 1) \
                           .when(col("comment").like("%I have a cat%"), 1) \
                           .when(col("comment").like("%my puppy%"), 1) \
                           .when(col("comment").like("%my pup%"), 1) \
                           .when(col("comment").like("%my kitty%"), 1) \
                           .when(col("comment").like("%my pussy%"), 1) \
                           .otherwise(0)))
df_clean.show()
# 1. Data preprocesing and build the classifier
from pyspark.ml.feature import RegexTokenizer, Word2Vec
from pyspark.ml.classification import LogisticRegression

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="words", pattern="\\W")

word2Vec = Word2Vec(inputCol="words", outputCol="features")
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[regexTokenizer, word2Vec])

# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(df_clean)
dataset = pipelineFit.transform(df_clean)
dataset.show()
# Remove the emtpy features caused by none English statements. 
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf

isnotEmpty = udf(lambda x: len(x) != 0, BooleanType())
dataset_noEmpty = dataset.filter(isnotEmpty('words'))

dataset_noEmpty.show()
(lable0_train,lable0_test)=dataset_noEmpty.filter(col('label')==1).randomSplit([0.7, 0.3],seed = 100)
(lable1_train, lable1_ex)=dataset_noEmpty.filter(col('label')==0).randomSplit([0.005, 0.995],seed = 100)
(lable1_test, lable1_ex2)=lable1_ex.randomSplit([0.002, 0.998],seed = 100)
trainingData = lable0_train.union(lable1_train)
testData=lable0_test.union(lable1_test)
print("Dataset Count: " + str(dataset.count()))
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))
#Build your ML model
#LogisticRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr = LogisticRegression(maxIter=10, regParam=0.3)

# Fit the model
lrModel = lr.fit(trainingData)

# Take a look at prediction on training set because we don't want to touch test samples.
# Cross valition and grid-search based finetuning will be applied later.
predictions = lrModel.transform(trainingData)
predictions.select('comment', 'features', 'rawPrediction', 'probability', 'prediction', 'label').show(10)

# Evaluate model using AUC = 0.945, which is good but also can be due to the overfitting
# We are going to go further with cross validation
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('AUC value on training samples: ' + '%.3f' % evaluator.evaluate(predictions))
# Model hyperparameter searching
#Parameter Tuning and K-fold cross-validation
#Note: The choice of hyperparameters is not optimal, especially the maxIter, owing to the running time concern.
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation.
# We want to finetune regParam and maxIter.
lr = LogisticRegression()
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.5, 2])
             .addGrid(lr.maxIter, [1, 2, 5])
             .build())

# Create 3-fold CrossValidator based on AUC.
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Run cross validations
cvModel_lr = cv.fit(trainingData)

# Use test set to measure the accuracy of our model on new data
predictions = cvModel_lr.transform(testData)

# Evaluate best model
print('AUC value of best Logistic Regression model on test samples: ' + '%.3f' % evaluator.evaluate(predictions))

# Display best hyper-parameters
print('Best regParam: ' + '%.2f' % cvModel_lr.bestModel._java_obj.getRegParam())
print('Best regParam: ' + str(cvModel_lr.bestModel._java_obj.getMaxIter()))

bestModel_lr = cvModel_lr.bestModel
#Try random forest model
#RandomForest
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import RandomForestClassifier

# Create ParamGrid for Cross Validation.
# We want to finetune maxDepth, maxBins and numTrees.
rf = RandomForestClassifier()
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 10])
             .build())

# Create 3-fold CrossValidator based on AUC.
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Run cross validations
cvModel_rf = cv.fit(trainingData)

# Use test set to measure the accuracy of our model on new data
predictions = cvModel_rf.transform(testData)

# Evaluate best model
print('AUC value of best RandomForest model on test samples: ' + '%.3f' % evaluator.evaluate(predictions))

# Display best hyper-parameters
print('Best maxDepth: ' + '%.2f' % cvModel_rf.bestModel._java_obj.getMaxDepth())
print('Best maxBins: ' + str(cvModel_rf.bestModel._java_obj.getMaxBins()))
print('Best numTrees: ' + str(cvModel_rf.bestModel._java_obj.getNumTrees()))

bestModel_rf = cvModel_rf.bestModel
# try GDBT
# Gradient boosting
rom pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import GBTClassifier

# Create ParamGrid for Cross Validation.
# We want to finetune maxDepth, maxBins and maxIter.
gdbt = GBTClassifier()
paramGrid = (ParamGridBuilder()
             .addGrid(gdbt.maxDepth, [2, 4])
             .addGrid(gdbt.maxBins, [20, 60])
             .addGrid(gdbt.maxIter, [5, 10])
             .build())

# Create 3-fold CrossValidator based on AUC.
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
cv = CrossValidator(estimator=gdbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Run cross validations
cvModel_gdbt = cv.fit(trainingData)

# Use test set to measure the accuracy of our model on new data
predictions = cvModel_gdbt.transform(testData)

# Evaluate best model
print('AUC value of best GDBT model on test samples: ' + '%.3f' % evaluator.evaluate(predictions))

# Display best hyper-parameters
#print('Best maxDepth: ' + str(cvModel_gdbt.bestModel._java_obj.getMaxDepth()))
#print('Best maxBins: ' + str(cvModel_gdbt.bestModel._java_obj.getMaxBins()))
#print('Best maxIter: ' + str(cvModel_gdbt.bestModel._java_obj.getMaxIter()))

bestModel_gdbt = cvModel_gdbt.bestModel
print('AUC value of best GDBT model on test samples: ' + '%.3f' % evaluator.evaluate(predictions))
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import GBTClassifier

# Create ParamGrid for Cross Validation.
# We want to finetune maxDepth, maxBins and maxIter.
gdbt = GBTClassifier()
paramGrid = (ParamGridBuilder()
             .addGrid(gdbt.maxDepth, [2, 4])
             .addGrid(gdbt.maxBins, [20, 60])
             .addGrid(gdbt.maxIter, [5, 10])
             .build())

# Create 3-fold CrossValidator based on AUC.
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
cv = CrossValidator(estimator=gdbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Run cross validations
cvModel_gdbt = cv.fit(trainingData)

# Use test set to measure the accuracy of our model on new data
predictions = cvModel_gdbt.transform(testData)

# Evaluate best model
#print('AUC value of best GDBT model on test samples: ' + '%.3f' % evaluator.evaluate(predictions))

# Display best hyper-parameters
#print('Best maxDepth: ' + str(cvModel_gdbt.bestModel._java_obj.getMaxDepth()))
#print('Best maxBins: ' + str(cvModel_gdbt.bestModel._java_obj.getMaxBins()))
#print('Best maxIter: ' + str(cvModel_gdbt.bestModel._java_obj.getMaxIter()))

bestModel_gdbt = cvModel_gdbt.bestModel
#Get the best model with best hyper-parameter
# According to the AUC result on test samples, GDBT with maxDepth=4, maxBins=20, and maxIter=10, is the best model.
best_model = bestModel_gdbt
#Apply the best model
# 2 Classify all the users
# Predict over all comments
predictions_over_comments = best_model.transform(dataset_noEmpty)

# Predict over all users. If a user has more than one comments, he or she has more than one prediction.
# We assume that we want to find the potential buyer so we don't want to miss any candidates.
# As a result, we apply max-win algorithm, which mean unless all prediction is 0, the user is marked as 1.
from pyspark.sql import functions as F
predictions_over_users = predictions_over_comments.groupBy('userid').agg(F.max('prediction').alias('predictions_over_users'))
predictions_over_users.show(5)

# Display the percetage of cat or dog owner.
#print('%.2f% of users are cat or dog owner.' % (predictions_over_users.filter(F.col('predictions_over_users') == 1).count()/predictions_over_users.count()*100))
print(predictions_over_users.filter(F.col('predictions_over_users') == 1).count()/predictions_over_users.count()*100)
#investigate the reasons from the text
# 3 get insight of users
# First, select cat or dog owners from the dataset
cat_dog_owner = ((predictions_over_users.filter(F.col('predictions_over_users') == 1)).join(predictions_over_comments, ['userid'])).select('userid', 'comment', 'words','predictions_over_users','creator_name')
# Second, find top 10 popular words in cat and dot owners' comments.
# In particular, common words, such as 'and', 'I', 'you', and 'we', have been kicked out.
common_words = ['i', 'the', 'and', 'a', 'to', 'you', 'is', 'it', 'of', 'my',
               'that', 'in', 'so', 'for', 'have', 'this', 'your', 'are',
               'was', 'on', 'with', 'but', 'he', 'they', 'be', 'me',
               'just', 'do', 'all', 'one', 'not', 'what', 'im', 'if',
               'get', 'when', 'them', 'its', 'she', 'would', 'can',
               'her', 'at', 'or', 'how', 'as', 'up', 'out', 'him',
               'dont', 'we', 'from', 'about', 'will', 'see', 'his',
               'great', 'there', 'know', 'had', 'really', 'people',
               'because', 'much', 'an', 'lol', 'got', 'more', 'some',
               'want', 'no', 'think', 'videos', 'has', 'very', 'now',
               'u', 'go', 'too', 'day', 'these', 'who', 'little',
               'did', 'by', 'their', 'could', 'make', 'been', 'hope',
               '3', 'should', 'also', 'am', 'always', 'why', 'keep',
               'were', 'well', 'those', 'then' ,'going', 'never',
               'thats', 'cant', 'only', 'new', 'way', 'other', 'look',
               'need', 'please', 'take', 'first']
  popular_words = cat_dog_owner.withColumn('word', F.explode(F.col('words'))).filter(~F.col('word').isin(common_words)).groupBy('word').count().sort('count', ascending=False)
  popular_words.show(10)
  # 4. Identify creators with cat and dog owners in the  text
  # Display the top 10 creator, who has the largest amount of cat and dog owner comments
creators = cat_dog_owner.groupBy('creator_name').count().sort('count', ascending=False)
creators.show(10)
# 5. Analysis and Future work
#In this project, we aim to build a model to identify cat or dog owners based on the comments for youtube videos related to animials or pets and then we also try to find out the topics interest them mostly.

#Totally, we have more than 5 million samples and we first remove the samples with no comments or with non-Enlish comments and we also label a comment based on if it contains sub-sentence like 'I have a pet' or 'my dog'.

#In the following, we finetune and select the model among logistic regression, random forest, and gradient boosting using cross-validation according to the area under the ROC curve (AUC). Finally, gradient boosting provides the best AUC value (i.e., 0.939). With the selected model, we #classify all the users and also extract insights about cat and dog owners and find topics important to cat and dog owners.

#In the future work, we can further optimized the model when more computation source is available.
