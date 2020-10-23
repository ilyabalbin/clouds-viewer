import io

import cv2
from PIL import Image
from flask import Flask, make_response, jsonify, request
import base64


from flask_cors import CORS

from utils import get_model, make_predict

app = Flask(__name__)
CORS(app)

print(1)
model = get_model()
print(2)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/getmask/', methods=['GET', 'POST'])
def get_mask():
    print('getmask')
    label = request.json['label_id']
    img_str = request.json['img']
    with open('test.png', 'wb') as f:
        f.write(base64.decodebytes(eval(f"b'{img_str}'")))

    image_np = cv2.imread('test.png')

    in_size = image_np.shape[:2]
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    mask = make_predict(image_np, model, label)
    mask = cv2.resize(mask, in_size)
    print(mask, mask.shape)

    pil_mask = Image.fromarray(mask)
    buff = io.BytesIO()
    pil_mask.convert('RGB').save(buff, format="JPEG")
    mask_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return make_response(jsonify({'mask': mask_str}), 200)


if __name__ == '__main__':
    app.run()
