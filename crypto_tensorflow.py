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
import glob
import os
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import threading
import time

# get the lastest file data
list_of_files = glob.glob('/mnt/c/Users/yuche/all_data/*')
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
result = read(latest_file)
data_frame = result.reverse()

# too large, only pick subset of data
data_frame = data_frame.tail_pct(0.02)

# train, test data split 70% train, 30% test
train_dh = data_frame.head_pct(0.7)
test_dh = data_frame.tail_pct(0.3)


# define the model
n_input = 3
n_features = 1
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# This function gathers data from a table into a NumPy ndarray
def table_to_numpy_double(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.double)

# A function to fit our LSTM model
def train_model(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    generator = TimeseriesGenerator(data, data, length=n_input, batch_size=1)
    model.fit(generator, epochs = 50)

# Train the LSTM model
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

# predict with the test set
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
