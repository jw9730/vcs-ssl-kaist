import json
import base64
import cv2
import numpy as np
from PIL import Image

from inpaint import inpaint


class Service:
    task = [
        {
            'name':"part3_kaist",
            'description':'[part 3] example task'
        }
    ]

    def __init__(self):
        self.service_model = ExampleModel()

    @classmethod
    def get_task_name(cls):
        return json.dumps(cls.task), 200

    def run(self, content):
        try:
            ret = self.service_model.run(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class ExampleModel(object):
    def __init__(self):
        pass

    def run(self, content):
        # input data
        img = self.stringToImage(content['img'])
        masked_img = self.stringToImage(content['masked_img'])
        json_sample_data = json_data = content['json_data']

        # output data
        scenegraph = inpaint(img, masked_img, json_data, json_sample_data)

        if scenegraph is None:
            return {
                'error':"invalid query"
            }
        return scenegraph

    def imageToString(self, img):
        return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

    def stringToImage(self, imagestring):
        data = base64.b64decode(imagestring)
        jpg_arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)


if __name__ == '__main__':
    img_file = '../example/n02090721_6873.jpg'
    masked_img_file = '../example/n02090721_6873.png'
    json_file = '../example/n02090721_6873.json'
    json_sample_file = '../example/n02090721_6873_1.json'

    # parse image and masked image
    img = Image.open(img_file)
    masked_img = Image.open(masked_img_file)

    # parse scene graph
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    with open(json_sample_file, 'r', encoding='utf-8') as f:
        json_sample_data = json.load(f)

    # predict
    json_data = json_sample_data
    output_data = inpaint(img, masked_img, json_data, json_sample_data)

    # save as json file
    with open('../example/output.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
