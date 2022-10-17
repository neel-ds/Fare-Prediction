#!/usr/bin/env python
# coding: utf-8

# ## Import the libraries

# In[ ]:


import warnings
import numpy as np                                            #Exploratory Data Analysis
import pyspark.pandas as ps
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import iplot
from random import random
from pyspark.ml import Pipeline                                #Modeling
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow                                                  #Deployment
import mlflow.sklearn
import mlflow.azureml
import azureml
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ### Mount the Azure Storage

# In[ ]:


dbutils.fs.mount(
  source = "wasbs://dataset@pricedatastorage.blob.core.windows.net",
  mount_point = "/mnt/data",
  extra_configs = {"fs.azure.account.key.pricedatastorage.blob.core.windows.net":"UUpMhyQ6jNC4fFy2QbYKaomV+hJAXB3P1oSxDP9jQyXIpBc1ljzZb4wAZQ4U7eyTYphlEHso+4tj+AStbx1GPQ=="})


# **Import the dataset**

# In[ ]:


df = spark.read.csv('/mnt/data/train.csv',inferSchema=True,header=True)
test_df = spark.read.csv('/mnt/data/test.csv',inferSchema=True,header=True)


# **Convert spark dataframe into pandas dataframe**

# In[ ]:


df = df.to_pandas_on_spark()
df = df.to_pandas()


# In[ ]:


test_df = test_df.to_pandas_on_spark()
test_df = test_df.to_pandas()


# In[ ]:


df.display()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# **Convert object datatype to datetime64[ns]**

# In[ ]:


df.key = df.key.astype('datetime64[ns]')
df.pickup_datetime = df.pickup_datetime.astype('datetime64[ns]')
test_df.key = test_df.key.astype('datetime64[ns]')
test_df.pickup_datetime = test_df.pickup_datetime.astype('datetime64[ns]')


# **Check and drop the null values**

# In[ ]:


df.isna().sum().sort_values(ascending=False)


# In[ ]:


df.dropna(inplace=True)


# **Check and drop the records which have negative fare amount**

# In[ ]:


df[df.fare_amount < 0].fare_amount


# In[ ]:


df.drop(df[df.fare_amount < 0].index, axis = 0, inplace = True)


# **Check and drop the records which have passenger count more than 6**

# In[ ]:


df[df.passenger_count > 6]


# In[ ]:


df.drop(df[df.passenger_count > 6].index, axis = 0, inplace = True)


# **Check the standard range of latitude and longitude, if value bound the range then drop**

# In[ ]:


df.pickup_latitude.between(-90,90,inclusive=True).sum()


# In[ ]:


df.drop(df[df.pickup_latitude < -90].index, axis = 0, inplace = True) 
df.drop(df[df.pickup_latitude > 90].index, axis = 0, inplace = True)


# In[ ]:


df.dropoff_latitude.between(-90,90,inclusive=True).sum()
df.drop(df[df.dropoff_latitude < -90].index, axis = 0, inplace = True) 
df.drop(df[df.dropoff_latitude > 90].index, axis = 0, inplace = True)


# In[ ]:


df.pickup_longitude.between(-180,180,inclusive=True).sum()
df.drop(df[df.pickup_longitude < -180].index, axis = 0, inplace = True) 
df.drop(df[df.pickup_longitude > 180].index, axis = 0, inplace = True)


# In[ ]:


df.dropoff_longitude.between(-180,180,inclusive=True).sum()
df.drop(df[df.dropoff_longitude < -180].index, axis = 0, inplace = True) 
df.drop(df[df.dropoff_longitude > 180].index, axis = 0, inplace = True)


# **Function to calculate distance using haversine formula**
# - Distance is calculated in miles.

# In[ ]:


def distance(data, lat1, long1, lat2, long2):
    x = np.sin(np.radians(df[lat2]-df[lat1])/2.0) ** 2 + np.cos(np.radians(df[lat1])) * np.cos(np.radians(df[lat2])) * \
        np.sin(np.radians(df[long2]-df[long1])/2.0) ** 2
    y = 2 * np.arctan2(np.sqrt(x), np.sqrt(1 - x))
    data['distance'] = 3959 * y
    pass


# In[ ]:


distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# **Feature scaling of pickup datetime**

# In[ ]:


df['year'] = df.pickup_datetime.dt.year
df['month'] = df.pickup_datetime.dt.month
df['date'] = df.pickup_datetime.dt.day
df['day_of_week'] = df.pickup_datetime.dt.dayofweek
df['hour'] = df.pickup_datetime.dt.hour


# ## Data Visualization

# **Histogram of fare amount**

# In[ ]:


Hist = [go.Histogram(x=df.fare_amount[:1000000])]
layout = go.Layout(title = 'Fare Amount')
fig = go.Figure(data = Hist, layout = layout)
iplot(fig)


# **Scatterplot of passenger count and fare amount**

# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=df.passenger_count, y=df.fare_amount)
plt.xlabel('Passenger Count')
plt.ylabel('Fare')


# **Scatterplot of date and fare amount**

# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=df.date, y=df.fare_amount, s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')


# **Scatterplot of hour and fare**

# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=df.hour, y=df.fare_amount)
plt.xlabel('Hour')
plt.ylabel('Fare')


# **Scatterplot of fare amount vs day of week**

# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=df.day_of_week, y=df.fare_amount, s=1.5)
plt.xlabel('Day of Week')
plt.ylabel('Fare')


# **Relation between distance and fare**

# In[ ]:


plt.figure(figsize=(16,10))
plt.scatter(x=df.distance, y=df.fare_amount, s=1.5)
plt.xlabel('Distance')
plt.ylabel('Fare')


# **Distribution of distance relative to fare and linear patterns in the distance**

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(18,7))
axs[0].scatter(df.distance, df.fare_amount, alpha=0.2)
axs[0].set_xlabel('Distance')
axs[0].set_ylabel('Fare')
axs[0].set_title('All data')

idx = (df.distance < 10) & (df.fare_amount < 75)
axs[1].scatter(df[idx].distance, df[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('Distance')
axs[1].set_ylabel('Fare')
axs[1].set_title('Distance < 10 mile, fare < $75');


# ### Feature Engineering

# **Scenario**
# 
# - Pickup latitude and longitude are 0, Dropoff latitude and longitude are not 0 but fare amount is 0
# 
# - vice versa.

# In[ ]:


df.drop(df.loc[((df.pickup_latitude == 0) & (df.pickup_longitude == 0))&((df.dropoff_latitude != 0) & (df.dropoff_longitude != 0)) & (df.fare_amount == 0)].index, axis = 0, inplace = True)


# In[ ]:


df.drop(df.loc[((df.pickup_latitude != 0) & (df.pickup_longitude != 0))&((df.dropoff_latitude == 0) & (df.dropoff_longitude == 0)) & (df.fare_amount == 0)].index, axis = 0, inplace = True)


# **There are so many irrelevant records in the correlation of distance and fare.**
# 
# So as per the statistics of the data available for the new york city, base fare for the taxi ride is `$`2.5 and average `$`1.56 per mile. 
# > Fare = (Distance `*` 1.56) + 2.5

# **Check the records if the distance is above 200 and fare amount is not equal to 0**

# In[ ]:


distance_check = df.loc[(df.distance > 200)&(df.fare_amount != 0)]
distance_check


# **Update all the records with correct values if satisfy the above condition**

# In[ ]:


distance_check.distance = distance_check.apply(lambda x: (x.fare_amount - 2.50) / 1.56, axis = 1)
df.update(distance_check)


# **Check if both distance and fare are 0 then it should be dropped**

# In[ ]:


df[(df.distance == 0) & (df.fare_amount == 0)]


# In[ ]:


df.drop(df[(df.distance == 0) & (df.fare_amount == 0)].index, axis = 0, inplace = True)


# **Check the fare amount is less than the base amount in the working days then drop it**

# In[ ]:


df.loc[(((df.hour >= 6) & (df.hour <= 20)) & ((df.day_of_week >= 1) & (df.day_of_week <= 5)) & (df.distance == 0) & (df.fare_amount < 2.5))]


# In[ ]:


df.drop(df.loc[(((df.hour >= 6) & (df.hour <= 20)) & ((df.day_of_week >= 1) & (df.day_of_week <= 5)) & 
                  (df.distance == 0) & (df.fare_amount < 2.5))].index, axis = 0, inplace = True)


# **Check and drop the records if fare is less than base amount on the weekends**

# In[ ]:


df.loc[((df.day_of_week == 0) | (df.day_of_week == 6)) & (df.distance == 0) & (df.fare_amount < 2.5)]


# In[ ]:


df.drop(df.loc[((df.day_of_week == 0) | (df.day_of_week == 6)) & (df.distance == 0) & (df.fare_amount < 2.5)].index, axis = 0, inplace = True)


# **Check if fare is 0 but distance is not 0 then impute it with mathematical calculation**

# In[ ]:


fare_check = df.loc[(df.fare_amount == 0) & (df.distance != 0)]
fare_check.sort_values('distance',ascending = False)


# In[ ]:


fare_check.fare_amount = fare_check.apply(lambda x: ((x.distance * 1.56) + 2.5), axis = 1)
df.update(fare_check)


# **Check if fare is above 3 but distance is equal to 0 then correct the distance according to the fare**

# In[ ]:


dist_check = df.loc[(df.fare_amount > 3) & (df.distance == 0)]
dist_check.sort_values('fare_amount',ascending = False)


# In[ ]:


dist_check.distance = dist_check.apply(lambda x: ((x.distance - 2.5) / 1.56), axis = 1)
df.update(dist_check)


# **Check if distance is above 100 but fare is still lower than 100, then fare should be imputed with correct values**

# In[ ]:


target_check = df.loc[(df.fare_amount < 100) & (df.distance > 100)]
target_check.sort_values(['distance','fare_amount'], ascending = False)


# In[ ]:


target_check.fare_amount = target_check.apply(lambda x: ((x.distance * 1.56) + 2.5), axis = 1)
df.update(target_check)


# **After all corrections, correlation of distance and fare amount should be linear and fair**

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(16,10))
ax.scatter(x=df.distance, y=df.fare_amount, s=1.5)
ax.set_xlabel('Distance')
ax.set_ylabel('Fare')
ax.set_title('Fare Distribution with relation to distance')


# **Get the minimum and maximum range of the co-ordinates from the test dataset to plot the graph**

# In[ ]:


bb = (min(test_df.pickup_longitude.min(), test_df.dropoff_longitude.min()), 
      max(test_df.pickup_longitude.max(), test_df.dropoff_longitude.max()), 
      min(test_df.pickup_latitude.min(), test_df.dropoff_latitude.min()), 
      max(test_df.pickup_latitude.max(), test_df.dropoff_latitude.max()))
bb


# **Function filter the range of pickup and dropoff locations from the training dataset**

# In[ ]:


def filter_range(df, bb):
    return (df.pickup_longitude >= bb[0]) & (df.pickup_longitude <= bb[1]) & \
           (df.pickup_latitude >= bb[2]) & (df.pickup_latitude <= bb[3]) & \
           (df.dropoff_longitude >= bb[0]) & (df.dropoff_longitude <= bb[1]) & \
           (df.dropoff_latitude >= bb[2]) & (df.dropoff_latitude <= bb[3])


# **Plot of non-linear traffic in the city**

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(24,12))
idx = filter_range(df, (-74.1, -73.7, 40.6, 40.9))
axs[0].set_title('Pickup Locations')
axs[0].scatter(df[idx].pickup_longitude, df[idx].pickup_latitude, c='r', s=0.01, alpha=0.5)
axs[1].set_title('Dropoff Locations')
axs[1].scatter(df[idx].dropoff_longitude, df[idx].dropoff_latitude, c='b', s=0.01, alpha=0.5)


# **Correlation map of features**

# In[ ]:


plt.figure(figsize=(15,12))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# **Prepare the test dataset**

# In[ ]:


distance(test_df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# In[ ]:


test_df['year'] = test_df.pickup_datetime.dt.year
test_df['month'] = test_df.pickup_datetime.dt.month
test_df['date'] = test_df.pickup_datetime.dt.day
test_df['day_of_week'] = test_df.pickup_datetime.dt.dayofweek
test_df['hour'] = test_df.pickup_datetime.dt.hour


# **Drop the insignificant variables**

# In[ ]:


df.drop(['key','pickup_datetime'], axis = 1, inplace = True)
test_df.drop(['key','pickup_datetime'], axis = 1, inplace = True)


# **Convert pandas dataframe to spark dataframe for further modeling**

# In[ ]:


train_df = ps.from_pandas(df)
train_df = train_df.to_spark()


# ## Modeling

# **Linear Regression**

# In[ ]:


feature_assembler = VectorAssembler(inputCols=['pickup_longitude', 'dropoff_longitude', 'passenger_count', 'distance', 'year', 'month', 'date'], outputCol="features")

lr = LinearRegression(labelCol="fare_amount")

pipeline = Pipeline(stages=[feature_assembler, lr])

train, test = train_df.randomSplit([0.75, 0.25])

lr_model = pipeline.fit(train)

predictions = lr_model.transform(test)

evaluator = RegressionEvaluator(labelCol= 'fare_amount', predictionCol= 'prediction')

print('RMSE:', evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
print('R-squared:', evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))


# **Decision Tree**

# In[ ]:


feature_assembler = VectorAssembler(inputCols=['pickup_longitude', 'dropoff_longitude', 'passenger_count', 'distance', 'year', 'month', 'day_of_week', 'hour'], outputCol="features")

dt = DecisionTreeRegressor(labelCol="fare_amount")

pipeline = Pipeline(stages=[feature_assembler, dt])

train, test = train_df.randomSplit([0.75, 0.25])

dt_model = pipeline.fit(train)

predictions = dt_model.transform(test)

evaluator = RegressionEvaluator(labelCol= 'fare_amount', predictionCol= 'prediction')

print('RMSE:', evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
print('R-squared:', evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))


# **Random Forest**

# In[ ]:


feature_assembler = VectorAssembler(inputCols=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'year', 'month', 'date', 'hour'], outputCol="features")

rf = RandomForestRegressor(labelCol="fare_amount")

pipeline = Pipeline(stages=[feature_assembler, rf])

train, test = train_df.randomSplit([0.75, 0.25])

rf_model = pipeline.fit(train)

predictions = rf_model.transform(test)

evaluator = RegressionEvaluator(labelCol= 'fare_amount', predictionCol= 'prediction')

print('RMSE:', evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
print('R-squared:', evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))


# **Gradient Boosting**

# In[ ]:


feature_assembler = VectorAssembler(inputCols=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'year', 'month', 'date', 'day_of_week'], outputCol="features")

gbr = GBTRegressor(labelCol="fare_amount")

pipeline = Pipeline(stages=[feature_assembler, gbr])

train, test = train_df.randomSplit([0.75, 0.25])

gbr_model = pipeline.fit(train)

predictions = gbr_model.transform(test)

evaluator = RegressionEvaluator(labelCol= 'fare_amount', predictionCol= 'prediction')

print('RMSE:', evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
print('R-squared:', evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))


# ## Deployment

# **Create or load an Azure ML workspace**
# > It will load a workspace or create a new one if it does not exist.

# In[ ]:


workspace_name = "fare-prediction"
workspace_location = "eastus"
resource_group = "NYC_Price_Prediction"
subscription_id = "75b2e7f9-9c16-4ece-9685-e7db922c1b62"   
authentication = InteractiveLoginAuthentication(tenant_id = "bc7e12a8-5234-4447-9371-44fc05a1d39c")

workspace = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             auth = authentication,
                             exist_ok = True)


# **Train the best performance model on the dataset using MLflow to log metrics, parameters, artifacts and model**

# In[ ]:


def fare_training(df, alpha, n_estimators):
    warnings.filterwarnings("ignore")
    if float(alpha) is None:
        alpha = 0.9
    else:
        alpha = float(alpha)
    
    if int(n_estimators) is None:
        n_estimators = 100
    else:
        n_estimators = int(n_estimators)
        
    np.random.seed(42)
    train, test = train_test_split(df)
    X_train = train.drop(["fare_amount"], axis=1)
    X_test = test.drop(["fare_amount"], axis=1)
    y_train = train[["fare_amount"]]
    y_test = test[["fare_amount"]]

    def evaluation_metrics(true, pred):
        return np.sqrt(mean_squared_error(true, pred)), mean_absolute_error(true, pred), r2_score(true, pred)
  
    with mlflow.start_run() as run:
        gbr = GradientBoostingRegressor(alpha=alpha, n_estimators = n_estimators, random_state = 42)
        gbr.fit(X_train, y_train)

        predictions = gbr.predict(X_test)

        (rmse, mae, r2) = evaluation_metrics(y_test, predictions)

        print("GradientBoostingRegressor(alpha=%f, n_estimators=%f):" % (alpha, n_estimators))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2 Score: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("n_estimators", n_estimators)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(gbr, "model")
        
        temp = int(random()*1000)
        path = "/dbfs/mlflow/fare_test/model-%f-%f-%f" % (alpha, n_estimators, temp)
        mlflow.sklearn.save_model(gbr, path)

        run_id = run.info.run_id
        print('Run ID: ', run_id)
        model_uri = "runs:/" + run_id + "/model"
        print('model_uri: ', model_uri)

    return run_id, model_uri


# In[ ]:


run_id, model_uri = fare_training(df, 0.01, 80)


# **Create a container image for trained model to deploy in Azure Container Instances(ACI) using MLflow**
# 
# Further as per the requirement container image can be deployed to ACI for staging which serve as REST endpoint and Azure Kubernetes Service(AKS) for production.

# In[ ]:


model_image, azure_model = mlflow.azureml.build_image(model_uri = model_uri, 
                                                      workspace = workspace,
                                                      model_name = "gbr-model",
                                                      image_name = "gbr-model",
                                                      description="Gradient Boosting Regressor for fare prediction",
                                                      synchronous=False)
model_image.wait_for_creation(show_output=True)

