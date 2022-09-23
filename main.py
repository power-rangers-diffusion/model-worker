import json
import os

import boto3
import torch

from torch import autocast
from diffusers import StableDiffusionPipeline

DEVICE = "cuda"
HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

def initialize_model_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=HUGGINGFACE_TOKEN
    )

    return pipe.to(DEVICE)

pipeline = initialize_model_pipeline()

sqs = boto3.resource("sqs")
queue = sqs.get_queue_by_name(QueueName="inference-request.fifo")

messages = queue.receive_messages(WaitTimeSeconds=20, MaxNumberOfMessages=1)
for message in messages:
    body = json.loads(message.body)

    with autocast("cuda"):
        pipeline(prompt=body["prompt"], strength=0.75, guidance_scale=7.5).images

    message.delete()