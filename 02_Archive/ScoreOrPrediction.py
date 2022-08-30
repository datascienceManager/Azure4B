# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import joblib
import json
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from azureml.core import Model,Workspace,Run


# %%
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'RegressorModel.pkl')
    model = joblib.load(model_path)

# %%
def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    ypredict = model.predict(data)
    return ypredict.to_list()


