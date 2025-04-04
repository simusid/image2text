from glob import glob
from PIL import Image
import numpy as np
import requests
import base64
import json
import cv2

# input an np array of an image
# returns b64 encoded
def nparray_to_base64_jpg(image):
   _, jpg_buffer = cv2.imencode('.jpg', image)
   return base64.b64encode(jpg_buffer.tobytes()).decode('utf-8')

def sendImage(img, url = "http://sdc:8896/upload"):
    start = time()
    b64_img = nparray_to_base64_jpg(img)
    
    response = requests.post(url, json={"image":b64_img})
    rc = json.loads(response.text)
    print(f"\nElapsed time: {time()-start} seconds")
    return rc['text'].split("assistant\n")[-1]
    
sendImage(img)
