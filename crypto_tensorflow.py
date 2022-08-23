# Connect to deephaven server
from deephaven_server import Server
s = Server(port=10000, jvm_args=["-Xmx4g"])
s.start()
# UGP lock is automatically acquired for each query operation
from deephaven import ugp
ugp.auto_locking = True

# Deephaven imports
from deephaven import dtypes as dht
from deephaven.learn import gather
from deephaven import learn
from deephaven.parquet import read
from deephaven import pandas as dhpd
from deephaven import new_table
from deephaven.column import string_col, int_col,double_col

# Python imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import numpy as np
import threading
import time
import glob
import os

# get the lastest file data
list_of_files = glob.glob('/mnt/c/Users/yuche/all_data/*')
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
result = read(latest_file)
data_frame = dhpd.to_pandas(result)
data_frame=data_frame.iloc[::-1]
# too large, only pick subset of data
data_size=int(len(data_frame)*0.98)
data_frame=data_frame.iloc[data_size:]
data_frame=data_frame.reset_index(drop=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
data_frame['Price'] = scaler.fit_transform(data_frame['Price'].values.reshape(-1,1))
train_size=int(len(data_frame)*0.7)
train_data=data_frame.iloc[:train_size]
test_data=data_frame.iloc[train_size:]
train_dh=dhpd.to_table(train_data)
test_dh=dhpd.to_table(test_data)


# set up input and feature size
n_input = 4
n_features = 1
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


def table_to_numpy_double(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.double)

def train_model(data):
    global model
    new_data=data.reshape(-1,1)
    generator = TimeseriesGenerator(new_data, new_data, length=n_input, batch_size=1)
    model.fit(generator, epochs = 50)

learn.learn(
    table = train_dh,
    model_func = train_model,
    inputs = [learn.Input("Price", table_to_numpy_double)],
    outputs = None,
    batch_size = train_dh.size
)

def get_predicted_class(data, idx):
    return data[idx]

new_data=train_data["Price"].values.reshape(-1,1)

def predict_with_model(data):
    test_predictions = []
    global new_data
    first_eval_batch = new_data[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(len(data)):
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]
        # # append the prediction into the array
        test_predictions.append(current_pred[0]) 
        add_data=data[i]
        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:,1:,:],[[add_data]],axis=1)
    return test_predictions

a=learn.learn(
    table = test_dh,
    model_func = predict_with_model,
    inputs = [learn.Input("Price", table_to_numpy_double)],
    outputs = [learn.Output("Predicted_price", get_predicted_class, "double")],
    batch_size = test_dh.size
)

table_pd=dhpd.to_pandas(a)
table_pd['Price'] = scaler.inverse_transform(table_pd['Price'].values.reshape(-1,1))
table_pd['Predicted_price']=scaler.inverse_transform(table_pd['Predicted_price'].values.reshape(-1,1))
a=dhpd.to_table(table_pd)
