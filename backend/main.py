from flask import Flask, g, render_template, request, json, jsonify, make_response
from flask_cors import CORS, cross_origin
from utils.database import get_database, get_item_map

import os

import pandas as pd
from predictor.rec_model import PopModel, TransferModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files'
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = TransferModel()#PopModel()
item_map = None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/get_test_predictions', methods=['POST'])
def get_test_predictions():
    incoming_file = request.files['dataset']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], incoming_file.filename)
    incoming_file.save(filepath)

    test_set = pd.read_csv(filepath, sep='\t')

    preds = model.predict_dataset(test_set).to_json(orient='records')

    return preds


@app.route('/predict_receipt', methods=['POST'])
def predict_receipt():
    sample = request.get_json()
    predict = model.predict_sample(sample)
    return str(predict)


@app.route('/train_model', methods=['POST'])
def train_model():
    incoming_file = request.files['dataset']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], incoming_file.filename)
    incoming_file.save(filepath)

    train_set = pd.read_csv(filepath, sep='\t')

    item_map = get_item_map(train_set)

    model.fit(train_set)

    return jsonify(item_map)


if __name__ == '__main__':
    app.run(debug=True)

