from deephaven_server import Server
s = Server(port=10000, jvm_args=["-Xmx4g"])
s.start()
# UGP lock is automatically acquired for each query operation
from deephaven import ugp
ugp.auto_locking = True


# Deephaven imports
from deephaven import DynamicTableWriter
from deephaven import dtypes as dht
from deephaven.learn import gather
from deephaven import read_csv
from deephaven import learn
from deephaven.parquet import read
from deephaven import pandas as dhpd
from deephaven import new_table
from deephaven.column import string_col, int_col,double_col

# Python imports
import numpy as np
import threading
import time
import torch
import random
import pandas as pd 
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# load the data
import glob
import os
list_of_files = glob.glob('/mnt/c/Users/yuche/all_data/*')
latest_file = max(list_of_files, key=os.path.getctime)
result = read(latest_file)
data_frame = dhpd.to_pandas(result)
data_frame=data_frame.iloc[::-1]
# only pick subset of data
data_size=int(len(data_frame)*0.98)
data_frame=data_frame.iloc[data_size:]
x_min=np.double(data_frame["Price"].min())
x_max=np.double(data_frame["Price"].max())
print(x_min)
print(x_max)
scaler = MinMaxScaler(feature_range=(-1, 1))
data_frame['Price'] = scaler.fit_transform(data_frame['Price'].values.reshape(-1,1))
data_frame=data_frame.reset_index(drop=True)
train_dh=dhpd.to_table(data_frame)





# train, test split
def data_split(crypto, look_back):
    data_raw = crypto # convert to numpy array
    data = []
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    data = np.array(data)
    x_train = data[:,:-1,:]
    y_train = data[:,-1,:]
    return [x_train, y_train]


#build the structure
# Build model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2 
output_dim = 1


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

def table_to_numpy_double(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.double)
    


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model=model.to(device)
loss_fn = torch.nn.MSELoss().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


look_back = 4
def train_model(data):
    x_train, y_train= data_split(data, look_back)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    x_train=x_train.to(device)
    y_train=y_train.to(device)
    # Train model
    #####################
    num_epochs = 100
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim =look_back-1
    for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()
    
    # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred.to(device), y_train.to(device))
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()
        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()


learn.learn(
    table = train_dh,
    model_func = train_model,
    inputs = [learn.Input("Price", table_to_numpy_double)],
    outputs = None,
    batch_size = train_dh.size
)




from websocket import create_connection, WebSocketConnectionClosedException
import json
ws = create_connection("wss://ws-feed.exchange.coinbase.com")
ws.send(
    json.dumps(
        {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["matches"],
        }
    )
)


from deephaven.time import to_datetime
from deephaven import DynamicTableWriter
import deephaven.dtypes as dht
from deephaven.time import to_datetime, lower_bin

from threading import Thread

def coinbase_time_to_datetime(strn):
    return to_datetime(strn[0:-1] + " UTC")

dtw_column_converter = {
    'price': float,
    'time': coinbase_time_to_datetime
}

dtw_columns = {
    'time': dht.DateTime,
    'price': dht.float_

}

dtw = DynamicTableWriter(dtw_columns)
time_dic={}
def thread_function():
    while True:
        try:
            data = json.loads(ws.recv())
            time=coinbase_time_to_datetime(data["time"])
            price=float(data["price"])
            time_mins=lower_bin(time, 60_000_000_000)
            if time_mins in time_dic:
                old_time=time_mins
                time_dic[time_mins][0]+=1
                time_dic[time_mins][1]+=price
            else:
                time_dic[time_mins]=[1,price]
                if len(time_dic)>1:
                    row_to_write = []
                    row_to_write.append(old_time)
                    row_to_write.append(time_dic[old_time][1]/time_dic[old_time][0])
                    dtw.write_row(*row_to_write)
        except Exception as e:
            print(e)

thread = Thread(target=thread_function)
thread.start()

coinbase_websocket_table = dtw.table


def get_predicted_class(data, idx):
    return data

First_time=0

def predict_with_model(data):
    global model
    global last_three_data
    global First_time
    data=data[0]
    value=((data[0]- x_min)/(x_max - x_min))* (1 - (-1)) + (-1)
    if First_time==0:
        last_three_data=np.array([value,value,value])
        last_three_data=last_three_data.reshape((1, 3, 1))
        First_time=1
        test_data = torch.from_numpy(last_three_data).type(torch.Tensor)
        test_data=test_data.to(device)
        y_test_pred = model(test_data)
        y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
        y_test_pred=y_test_pred.reshape(1,-1)[0]
    else:
        add_data=np.array([value])
        last_three_data = np.append(last_three_data[:,1:,:],[[add_data]],axis=1)
        test_data = torch.from_numpy(last_three_data).type(torch.Tensor)
        test_data=test_data.to(device)
        y_test_pred = model(test_data)
        y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
        y_test_pred=y_test_pred.reshape(1,-1)[0]
    return y_test_pred[0]





time.sleep(60)
real_time_prediction = learn.learn(
    table = coinbase_websocket_table,
    model_func = predict_with_model,
    inputs = [learn.Input("price", table_to_numpy_double)],
    outputs = [learn.Output("Predicted_price", get_predicted_class, "double")],
    batch_size = 1
)

from deephaven.plot.figure import Figure
figure = Figure()
plot_single = figure.plot_xy(series_name="price", t=real_time_prediction, x="time", y="price").plot_xy(series_name="Predicted_price", t=real_time_prediction, x="time", y="Predicted_price")
plot1 = plot_single.show()










    

