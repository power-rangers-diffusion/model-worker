import json
import os
import base64
from io import BytesIO, StringIO
import time

import requests
import boto3
import torch

from torch import autocast
from diffusers import StableDiffusionPipeline

DEVICE = "cuda"
HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

def main():
    def initialize_model_pipeline():
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=HUGGINGFACE_TOKEN
        )

        return pipe.to(DEVICE)

    pipeline = initialize_model_pipeline()

    sqs = boto3.resource("sqs", region_name='us-east-1')
    queue = sqs.get_queue_by_name(QueueName="inference-request.fifo")

    while True:
        messages = queue.receive_messages(WaitTimeSeconds=20, MaxNumberOfMessages=1)
        for message in messages:
            body = json.loads(message.body)
            message_id = message.message_id

            with autocast("cuda"):
                images = pipeline(prompt=body["prompt"], strength=0.75, guidance_scale=7.5).images
                similarity_results = get_similar_images(body["prompt"])
                upload_to_s3(message_id, images[0], similarity_results)

            message.delete()
        time.sleep(5)

# Find similar images to text prompt

SEARCH_URL = 'https://knn5.laion.ai/knn-service'
#SEARCH_URL = 'http://localhost:9753/knn-service'

def get_similar_images(prompt):
    headers = {
        'Content-Type': 'text/plain;charset=UTF-8',
        'Accept': '*/*',
    }

    data = {
        "text": prompt,
        "image": None,
        "image_url": None,
        "embedding_input": None,
        "modality": "image",
        "num_images": 10,
        "indice_name": "laion_400m",
        "num_result_ids": 3000,
        "use_mclip": False,
        "deduplicate": True,
        "use_safety_model": True,
        "use_violence_detector": True,
        "aesthetic_score": "9",
        "aesthetic_weight": "0.5"
    }

    res = requests.post(SEARCH_URL, headers=headers, data=json.dumps(data)).json()
    return res[:data['num_images']]

# Upload images to S3

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

def upload_to_s3(message_id, image, similarity_results):
    def image_to_base64(image):
        '''
        image: PIL image
        returns image as a base 64 string
        '''
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('ascii')

    image_json = json.dumps({
        "message_id": message_id,
        "gen_image": image_to_base64(image),
        "similarity_results": similarity_results
    })

    # Upload the file to the s3 bucket
    json_buffer = StringIO(image_json)
    file_buffer = BytesIO(json_buffer.getvalue().encode())
    client = boto3.client('s3', region_name='ca-central-1', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    client.upload_fileobj(file_buffer, 'fsdl-52-images', message_id + '.json')

if __name__ == "__main__":
    main()