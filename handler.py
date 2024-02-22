import time
import requests
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from schemas.api import API_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA
from schemas.options import OPTIONS_SCHEMA
import io
import PIL
from PIL import Image
import base64

BASE_URL = 'http://127.0.0.1:7680'
TIMEOUT = 600

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.headers.update({"Content-Type": "application/json", 'Accept': 'application/json'})
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URL}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload):
    return session.post(
        url=f'{BASE_URL}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )


def validate_api(event):
    if 'api' not in event['input']:
        return {
            'errors': '"api" is a required field in the "input" payload'
        }

    api = event['input']['api']

    if type(api) is not dict:
        return {
            'errors': '"api" must be a dictionary containing "method" and "endpoint"'
        }

    api['endpoint'] = api['endpoint'].lstrip('/')

    return validate(api, API_SCHEMA)


def validate_payload(event):
    method = event['input']['api']['method']
    endpoint = event['input']['api']['endpoint']
    payload = event['input']['payload']
    validated_input = payload

    if endpoint == 'txt2img':
      validated_input = validate(payload, TXT2IMG_SCHEMA)
    elif endpoint == 'img2img':
      validated_input = validate(payload, IMG2IMG_SCHEMA)
    elif endpoint == 'options' and method == 'POST':
      validated_input = validate(payload, OPTIONS_SCHEMA)

    return endpoint, event['input']['api']['method'], validated_input



def pic_replace(payload, model):
  response_change_model = send_post_request('sdapi/v1/options', {"sd_model_checkpoint": model})
  # make auto sam by prompt
  sam_options = payload[0]
  response = send_post_request('sam/sam-predict', sam_options)
  result = response.json()
  logger.log(result, 'INFO')

  mask_string = result['masks'][2]

  with open('mask1.png', 'wb') as f:
    f.write(base64.b64decode(result['masks'][0]))
  with open('mask2.png', 'wb') as f:
    f.write(base64.b64decode(result['masks'][1]))
  with open('mask3.png', 'wb') as f:
    f.write(base64.b64decode(result['masks'][2]))
  # draw in mask
  input_image = payload[0]['input_image']
  
  imgdata = base64.b64decode(mask_string)
  im = Image.open(io.BytesIO(imgdata))
  width, height = im.size

  img2img_inpaint_options = payload[1]  
  img2img_inpaint_options['mask'] = mask_string
  img2img_inpaint_options['init_images'] = [input_image]
  # img2img_inpaint_options['width'] = width
  # img2img_inpaint_options['height'] = height
  response2 = send_post_request('sdapi/v1/img2img', img2img_inpaint_options)

  return response2


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #

def add_mask(payload):
  generate_mask_req = send_post_request('sam/sam-predict', {
    "sam_model_name": "sam_vit_h_4b8939.pth",
    "input_image": payload["init_images"][0],
    "sam_positive_points": [],
    "sam_negative_points": [],
    "dino_enabled": True,
    "dino_model_name": "GroundingDINO_SwinB (938MB)",
    "dino_text_prompt": "face",
    "dino_box_threshold": 0.4,
    "dino_preview_checkbox": False,
    "dino_preview_boxes_selection": [],
    "dilate_amount": 1
  })

  dilated_mask_req = send_post_request('sam/dilate-mask', {
    "input_image": payload["init_images"][0],
    "mask": generate_mask_req.json()['masks'][2],
    "dilate_amount": payload["dilate_amount"]
  })

  payload['mask'] = dilated_mask_req.json()['mask']

def handler(event):
    validated_api = validate_api(event)

    if 'errors' in validated_api:
        return {
            'error': validated_api['errors']
        }

    endpoint, method, validated_input = validate_payload(event)

    if 'errors' in validated_input:
        return {
            'error': validated_input['errors']
        }

    if 'validated_input' in validated_input:
        payload = validated_input['validated_input']
    else:
        payload = validated_input

    # set model
    send_post_request('sdapi/v1/options', {"sd_model_checkpoint": validated_api['validated_input']['model']})

    if "prompt" in payload.keys():
      if ("mask" not in payload.keys()) or ("mask" in payload.keys() and payload["mask"] == ""):
        add_mask(payload)

    try:
        logger.log(f'Sending {method} request to: /{endpoint}')

        if method == 'GET':
            response = send_get_request(endpoint)
        elif method == 'POST':
            response = send_post_request(endpoint, payload)

    except Exception as e:
        logger.log(e, 'INFO')
        return {
            'error': str(e)
        }

    if "dino_model_name" in payload.keys():
      dilated_mask = send_post_request('sam/dilate-mask', {
        "input_image": payload["input_image"],
        "mask": response.json()['masks'][2],
        "dilate_amount": payload["dilate_amount"] or 0
      })

      return {'images': [dilated_mask.json()['mask']]}

    if "prompt" in payload.keys():
      return {'images': [response.json()['images'][0]]}

    return response.json()


if __name__ == "__main__":
    wait_for_service(url=f'{BASE_URL}/sdapi/v1/sd-models')
    logger.log('Automatic1111 API is ready', 'INFO')
    logger.log('Starting RunPod Serverless...', 'INFO')
    runpod.serverless.start(
      {
        'handler': handler
      }
    )