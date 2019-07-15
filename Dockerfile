# Dockerfile of Example
# Version 1.0
# Base Images
FROM tensorflow/tensorflow:1.4.1-gpu-py3
#MAINTAINER
MAINTAINER Ironball
#RUN pip --no-cache-dir install Pillow==5.3.0
ADD c* /competition/checkpoints
ADD [^c]* /competition/ 
WORKDIR /competition


