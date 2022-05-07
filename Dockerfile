FROM python:3.9-alpine 
FROM "pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime"
COPY . /app
WORKDIR /app/code/
RUN pip install -r ../requirements.txt 
RUN pip install --no-cache-dir scipy matplotlib
CMD python ./main.py
