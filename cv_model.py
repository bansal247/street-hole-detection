from PIL import Image
from ultralytics import YOLO
import base64
from io import BytesIO
import numpy as np

class Model:
    def __init__(self):
        self.model = YOLO('best.pt')

    def predict(self, file):
        try:
            img = Image.open(BytesIO(file.read()))
            results = self.model(np.array(img), imgsz=640, device='cpu')
            im_array = results[0].plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save('result_temp.jpg')
            buffered = BytesIO()
            im.save(buffered, format="JPEG")
            byte_data = buffered.getvalue()
            base64_data = base64.b64encode(byte_data).decode('utf-8')
            return base64_data
        except Exception as e:
            print(e)
            return ''
