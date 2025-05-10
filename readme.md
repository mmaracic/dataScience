# Readme

Docker compose source (with added fixes)
https://github.com/umbraesoulsbane/anaconda

# Anaconda
Password to login in to jupyter notebooks is "password".

# Zeppelin and Spark

Spark source:
https://github.com/bitnami/bitnami-docker-spark
Zeppelin added and adjusted based on:
https://github.com/jgoodman8/zeppelin-spark-standalone-cluster

To enable Zeppelin to work with Spark, Spark executable and related files have to be copied into Zeppelin folder /spark. After Spark docker-compose has been started position into:
${dataScience_folder}/bitnami-docker-spark and execute:
docker cp datasci_spark_1:/opt/bitnami/spark .
Administrative privileges on docker are required. This will copy approx 0.5Gb of Spark files from Spark master container. The files represent Spark interpreter.

After logging into spark clickk username -> interpreters. There search for Spark interpreter and set variable zeppelin.spark.enableSupportedVersionCheck to false and save changes. If this is not done "This is not officially supported spark version" exception will appear when executing paragraphs. 