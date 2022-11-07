from flask import Flask, render_template, request, jsonify

import requests
import numpy as np
from io import BytesIO
from PIL import Image

from model import get_prediction

def url_to_img(url, save=True):
  img = Image.open(BytesIO(requests.get(url).content))
  if save:
      img.save('temp.jpg')
  return img

app = Flask(__name__)

@app.route('/')
def greeting():
    return "Skin detective API."

@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get('image_url')
    if url:
        pil_img = url_to_img(url)

    result = get_prediction(pil_img)
    print(result)
    
    return result


@app.route('/healthcheck')
def healthcheck():
    return "API is alive."

if __name__ == "__main__":
    app.run()

