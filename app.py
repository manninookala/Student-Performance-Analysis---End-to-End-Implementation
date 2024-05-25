import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)
regression_model = pickle.load(open('regmodel.pkl','rb'))
##encode = pickle.load(open('label_encoder.pkl','rb'))
scaler = pickle.load(open('scale.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    ##data = encode.fit_transform(data['data']['Extracurricular Activities'])

    data_transform = np.array(list(data.values())).reshape(1,-1)
    new_data = scaler.transform(data_transform)

    output = regression_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)




