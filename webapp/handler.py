import os
import pickle
import pandas as pd

from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load(open('model/model_xgb_tuned.pkl', 'rb'))

# Initialize API
app = Flask(__name__)


@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    # There is data
    if test_json:
        
        # Unique example
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        
        # Multiple example
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        # Instantiate Rossmann Class
        pipeline = Rossmann()
        
        # data cleaning
        dataset1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        dataset2 = pipeline.feature_engineering(dataset1)

        # data preparation
        dataset3 = pipeline.data_preparation(dataset2)

        # prediction
        dataset_response = pipeline.get_prediction(model, test_raw, dataset3)
        
        return dataset_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)