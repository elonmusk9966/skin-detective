# FastAPI
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import uvicorn


from typing import Union
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
    image_url: str = Field(default="", title="Image URL for skin diseases' detection.")


class Response_Item(BaseModel):
    result: dict = Field(default={}, title="Results for 3 classes (ez, ps, others).")

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
                            "result" : {"ez": 1.0, "ps": 0.0, "others": 0.0}
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

@app.get('/healthcheck')
def healthcheck():
    return "API is alive."

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1234)

