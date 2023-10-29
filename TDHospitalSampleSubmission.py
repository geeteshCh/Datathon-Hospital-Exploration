# Sample participant submission for testing
from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import random
from preproc import Dataset2
import numpy as np
from keras.models import load_model

app = Flask(__name__)


ds = Dataset2(pd.read_csv('./TD_HOSPITAL_TRAIN.csv'))
print(ds.data.shape)
print(ds.mean)
mean = ds.mean


class Solution:
    def __init__(self):
        #Initialize any global variables here
        self.model = load_model('example_basic.h5')
        print(self.model.summary())

    def calculate_death_prob(self, data_old):
        
        """
        This function should return your final prediction!
        """
        
        data = dict()
        for k, v in data_old.items():
            if v in ['nan', '']:
                data[k] = [np.nan]
            elif v[0].isdigit():
                data[k] = [float(v)]
            else:
                data[k] = [v]

        df = Dataset2(pd.DataFrame(data), True, mean).data

        print(df)

        print(df.shape)


        prediction = self.model.predict(df.to_numpy())
        
        return float(prediction[0][0])


# BOILERPLATE
@app.route("/death_probability", methods=["POST"])
def q1():
    solution = Solution()
    data = request.get_json()
    return {
        "probability": solution.calculate_death_prob(data)}


if __name__ == "__main__":
    


    app.run(host="0.0.0.0", port=5555)
