# FastAPI
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import uvicorn


from typing import Union
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from model import get_prediction, get_gradcam

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM


def url_to_img(url, save=True):
  img = Image.open(BytesIO(requests.get(url).content))
  if save:
    img.save('temp.jpg')
  return img


description = """
API to detect skin diseases.

"""

tags_metadata = [
    {
        "name": "predict",
        "description": """
    Get bot prediction for skin diseases. Supported types of skins:
        - ez: eczema 
        - ps: psoriasis
        - others: others' skin diseases
        """

    },
]

app = FastAPI(
    title='Skin Detective App',
    description=description,
    version='1',
    terms_of_service="http://example.com/terms/",
    contact={
            "name": "Vo Quoc Bang",
            "email": "bavo.imp@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=tags_metadata)


class Request_Item(BaseModel):
  image_url: str = Field(
      default="", title="Image URL for skin diseases' detection.")


class Request_Heatmap_Item(BaseModel):
  image_url: str = Field(
      default="", title="Image URL for skin diseases' detection.")
  class_name: str = Field(
      default="", title="Class name for skin diseases' detection heatmap.")


class Response_Item(BaseModel):
  result: dict = Field(
      default={}, title="Results for 3 classes (ez, ps, others).")


class Request_predict_heatmap_Item(BaseModel):
  image_url: str = Field(
      default="", title="Image URL for skin diseases' detection.")
  get_heatmap: bool = Field(
      default=False, title="Get heat map if skin diseases detected.")


@app.get('/')
def greeting():
  return "Skin detective API."


@app.post('/predict',
          response_model=Response_Item,
          responses={
              200: {
                  "description": "Results for query image.",
                  "content": {
                      "application/json": {
                          "example": {
                              "result": {"ez": 1.0, "ps": 0.0, "others": 0.0}
                          }
                      }
                  }
              }
          },

          )
def get_predict(
    item: Request_Item = Body(
        example={
            "image_url": "https://d3hcoe79thio2n.cloudfront.net/wp-content/uploads/2017/09/eczema-300x169.jpg"
        }
    ),
):
  url = item.dict().get('image_url', None)
  if url:
    pil_img = url_to_img(url)

  result = get_prediction(pil_img)
  print(result)
  return {"result": result}


@app.post('/heatmap',
          response_model=Response_Item,
          responses={
              200: {
                  "description": "Results base64 heat map for query jpg image.",
                  "content": {
                      "application/json": {
                          "example": {
                              "heatmap": "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"}
                      }
                  }
              }
          },

          )
def get_heatmap(
    item: Request_Heatmap_Item = Body(
        example={
            "image_url": "https://tradimec.com/wp-content/uploads/2021/04/benh-cham-eczema-2.jpg",
            "class_name": "ez"
        }
    ),
):
  url = item.dict().get('image_url', None)
  class_name = item.dict().get('class_name', None)
  if url:
    pil_img = url_to_img(url)

  result = get_gradcam(pil_img, class_name)
  return {"result": result}


@app.post('/predict_heatmap',
          response_model=Response_Item,
          responses={
              200: {
                  "description": "Results for query image.",
                  "content": {
                      "application/json": {
                          "example": {
                              "result": {"ez": 1.0, "ps": 0.0, "others": 0.0, "final_decision": "ez", "heatmap": "base64_jgp_image"},
                          }
                      }
                  }
              }
          },

          )
def get_predict_heatmap(
    item: Request_predict_heatmap_Item = Body(
        example={
            "image_url": "https://d3hcoe79thio2n.cloudfront.net/wp-content/uploads/2017/09/eczema-300x169.jpg",
            "get_heatmap": True
        }
    ),
):
  url = item.dict().get('image_url', None)
  return_heatmap = item.dict().get('get_heatmap', False)
  print(return_heatmap)
  if url:
    pil_img = url_to_img(url)

  result = get_prediction(pil_img)
  print(result)
  if return_heatmap and result["final_decision"] in ["ez", "ps"]:
    gradcam_result = get_gradcam(pil_img, result["final_decision"])
    result["heatmap"] = gradcam_result["heatmap"]
    return {"result": result}
  else:
    result["heatmap"] = ""
    return {"result": result}


@app.get('/healthcheck')
def healthcheck():
  return "API is alive."


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=1234)
