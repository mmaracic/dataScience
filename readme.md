# Readme

Docker compose source (with added fixes)
https://github.com/umbraesoulsbane/anaconda

# Anaconda
Password to login in to jupyter notebooks is "password".

## Pip libraries
### Main
* jupyter_contrib_nbextensions - Contains a collection of extensions that add functionality to the Jupyter notebook. These extensions are mostly written in Javascript, and are loaded locally in the browser. #Does not work with newer Jupyter versions
* jupyter_nbextensions_configurator - The jupyter_nbextensions_configurator jupyter server extension provides graphical user interfaces for configuring which nbextensions are enabled. #Does not work with newer Jupyter versions
* tensorflow
* scikit-learn
* scikit-image
* statsmodels - provides a complement to scipy for statistical computations including descriptive statistics and estimation and inference for statistical models.
* seaborn
* feature-engine - Feature-engine is a Python library with multiple transformers to engineer and select features for use in machine learning models.
* transformers
* torch
* accelerate - Run your *raw* PyTorch training script on any kind of device. Accelerate abstracts exactly and only the boilerplate code related to multi-GPUs/TPU/fp16 and leaves the rest of your code unchanged.
* protobuf - Protocol buffers are Google’s language-neutral, platform-neutral, extensible mechanism for serializing structured data – think XML, but smaller, faster, and simpler.
* sentencepiece - SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training
* gensim -  library for topic modelling, document indexing and similarity retrieval with large corpora

### Extra

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