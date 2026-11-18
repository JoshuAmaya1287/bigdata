
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session connected to Hive
spark = SparkSession.builder \
    .appName("Influencer ML Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# Step 2: Load data from your Hive table 'influencers_2026'
query = """
    SELECT posts, followers, avg_likes, eng_rate_60_day, influence_score 
    FROM influencers_2026
"""
influencers_df = spark.sql(query)

# Step 3: Handle null values by dropping them
influencers_df = influencers_df.na.drop()

# Step 4: Prepare features for MLlib
assembler = VectorAssembler(
    inputCols=["posts", "followers", "avg_likes", "eng_rate_60_day"],
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(influencers_df).select("features", "influence_score")

# Step 5: Split data into training (70%) and testing (30%) sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 6: Initialize and train the Linear Regression model
lr = LinearRegression(labelCol="influence_score")
lr_model = lr.fit(train_data)

# Step 7: Evaluate the model on the test data
test_results = lr_model.evaluate(test_data)

# Step 8: Print performance metrics
print(f"--- Influencer ML Model Results ---")
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with Happybase ----
metrics_data = [
    ('influencer_metric_1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('influencer_metric_1', 'cf:r2',   str(test_results.r2)),
]

def write_to_hbase_partition(partition):
    # Connect to HBase Thrift server running on the master node
    connection = happybase.Connection('master', port=9090)
    connection.open()
    table = connection.table('influencers_table_2026')
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

# Parallelize metrics and push to HBase via foreachPartition
rdd = spark.sparkContext.parallelize(metrics_data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop Spark session
spark.stop()
