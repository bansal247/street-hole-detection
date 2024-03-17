from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
from cv_model import Model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
@cross_origin()
def index():
    return jsonify(message='App is running')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Check if the request contains a file
    print('here', request.files)
    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty part without filename
    if file.filename == '':
        return 'No selected file', 400

    if file:

        result = model.predict(file=file)
        return jsonify(result=result)
    return 'some error occurred'


if __name__ == '__main__':
    model = Model()
    app.run(debug=True)
