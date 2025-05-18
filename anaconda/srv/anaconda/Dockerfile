FROM continuumio/anaconda3:2024.02-1

RUN apt-get update
RUN apt-get -qq -y install gcc
RUN apt-get -qq -y install g++
RUN pip install setuptools

ADD ./srv/anaconda/conf/requirements.txt /
RUN pip install -r requirements.txt

ADD ./srv/anaconda/conf/extra_requirements.txt /
RUN pip install -r extra_requirements.txt

RUN mkdir -p /root/.jupyter
COPY ./srv/anaconda/conf/jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json

EXPOSE 8888
