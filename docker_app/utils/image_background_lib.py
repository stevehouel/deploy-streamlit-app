import os
import boto3
import json
import base64
from io import BytesIO
from random import randint

#get a BytesIO object from file bytes
def get_bytesio_from_bytes(bytes):
    return BytesIO(bytes)

#get a base64-encoded string from file bytes
def get_base64_from_bytes(bytes):
    return base64.b64encode(get_bytesio_from_bytes(bytes).getvalue()).decode('utf-8')

#load the bytes from a file on disk
def load_bytes_from_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read()


# get the stringified request body for the InvokeModel API call
def get_titan_image_background_replacement_request_body(prompt, image_bytes, mask_prompt, negative_prompt=None, outpainting_mode = "DEFAULT"):
    input_image = get_base64_from_bytes(image_bytes)
    body = {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "image": input_image,
            "text": prompt,
            "maskPrompt": mask_prompt,
            "outPaintingMode": outpainting_mode,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": randint(0, 1000000),
        },
    }

    if negative_prompt:
        body["outPaintingParams"]["negativePrompt"] = negative_prompt

    return json.dumps(body)

# get a BytesIO object from the Titan Image Generator response
def get_titan_response_image(response):
    response = json.loads(response.get('body').read())
    images = response.get('images')
    image_data = base64.b64decode(images[0])
    return BytesIO(image_data)

# Generate an image using Amazon Titan Generator
def generate_titan_image(prompt, image_bytes, mask_prompt, negative_prompt=None, outpainting_mode = "DEFAULT"):
    bedrock = boto3.client('bedrock-runtime')
    body = get_titan_image_background_replacement_request_body(prompt, image_bytes, mask_prompt, negative_prompt, outpainting_mode)
    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-image-generator-v1",
        contentType="application/json",
        accept="application/json")

    output_image = get_titan_response_image(response)

    return output_image
