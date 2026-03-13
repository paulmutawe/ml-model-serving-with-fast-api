
import warnings
warnings.filterwarnings('ignore')

#from sympy import false
from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput
from scripts import s3
from fastapi import FastAPI, Request
import uvicorn
import os
import torch
import time
from transformers import AutoImageProcessor, pipeline

#model_ckpt = "google/vit-base-patch16-224-in21k"
#image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Donwload links

force_download = False

model_name = 'tinybert-sentiment-analysis/'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
    
sentiment_model = pipeline('text-classification', model=local_path, device=device)
    
model_name = 'tinybert-disaster-tweet/'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
    
tweeter_model = pipeline('text-classification', model=local_path, device=device)
    
model_name = 'vit-human-pose-classification/'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
    
    
image_processor = AutoImageProcessor.from_pretrained(local_path, use_fast=True)    
pose_model = pipeline('image-classification', model=local_path, device=device, image_processor=image_processor)

# Download ends

@app.get("/")
def read_root():
    return "Hello! I am up!!!"

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)
    
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]
    
    output = NLPDataOutput(model_name="tinybert-sentiment-analysis",
                           text = data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    
    return output

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = tweeter_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)
    
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]
    
    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text = data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    return output


@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    urls = [str(x) for x in data.url]
    start = time.time()
    output = pose_model(urls)
    end = time.time()
    prediction_time = int((end-start)*1000)
    
    labels = [x[0]['label'] for x in output]
    scores = [x[0]['score'] for x in output]
    
    output = ImageDataOutput(model_name="vit-human-pose-classification",
                           url = data.url,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    
    return output


if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8503, reload = True)
