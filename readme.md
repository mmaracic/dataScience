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
```
${dataScience_folder}/bitnami-docker-spark 
```
and execute:
```
docker cp datasci_spark_1:/opt/bitnami/spark .
```
Administrative privileges on docker are required. This will copy approx 0.5Gb of Spark files from Spark master container. The files represent Spark interpreter.

After logging into spark clickk username -> interpreters. There search for Spark interpreter and set variable zeppelin.spark.enableSupportedVersionCheck to false and save changes. If this is not done "This is not officially supported spark version" exception will appear when executing paragraphs. 

# Python 3 local environment
* Install Python3
* If python3 is not recognised as python command
```
sudo apt install python-is-python3
```
* Install pip (will not work if pip was installed using apt-get):
```
sudo apt install python3-pip
```
In MSYS:
```
pacman -S python3-pip
```
* Install possibility to create virtual environments
```
sudo apt install python3-venv
```
* Install hugginface cli
```
pip install -U "huggingface_hub[cli]"
```
* Add install folder to ~/.profile

For Windows
* Install Python 3
* Install pip:
```
python -m ensurepip --upgrade
```
```
python -m pip install --upgrade pip
```

* Add python and pip path <python_folder>\Scripts (e.g. C:\Python38\Scripts) to Windows path 

## PIP
The custom pip libraries need to be installed in virtual environment (otherwise for environment maintained for apt-get we will get - error: externally-managed-environment)

Check if virtual environment is active (if its not active it will write /usr/bin/python, if it is cmd will be prefixed by name of virtual env and this will print env path)
```
which python (Linux)
```
```
where python (Windows)
```
Create virtual environment in .venv subfolder
```
python3 -m venv .venv
```
Activate environment in .venv subfolder:
```
source .venv/bin/activate (Linux)
```
```
.\.venv\bin\activate (Windows)
```
In the virtual environment it will now be possible to install any needed libraries using:
```
python3 -m pip install requests
```
```
python3 -m pip install -r requirements.txt
```

To deactivate the current virtual environment use:
```  
deactivate
```
## Errors during installation of dependencies

Error:
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)>

### Solution1
https://stackoverflow.com/questions/64311305/ssl-certificate-verify-failed-error-while-using-pip
```
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"
```
### Solution2
Run
```  
python3 -m pip install -U certifi
```  
Then, run the following code to update your SSL certificate:
```  
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```  
