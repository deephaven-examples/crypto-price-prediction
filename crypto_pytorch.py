# connect to deephaven server
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
import glob
import os

# use the cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# load the data
list_of_files = glob.glob('/mnt/c/Users/yuche/all_data/*')
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
result = read(latest_file)
data_frame = dhpd.to_pandas(result)
data_frame=data_frame.iloc[::-1]

# only pick subset of data
data_size=int(len(data_frame)*0.98)
data_frame=data_frame.iloc[data_size:]
scaler = MinMaxScaler(feature_range=(-1, 1))
data_frame['Price'] = scaler.fit_transform(data_frame['Price'].values.reshape(-1,1))
data_frame=data_frame.reset_index(drop=True)
train_size=int(len(data_frame)*0.7)
train_data=data_frame.iloc[:train_size]
test_data=data_frame.iloc[train_size:]
train_dh=dhpd.to_table(train_data)
test_dh=dhpd.to_table(test_data)


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
    

# define the model 
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model=model.to(device)
loss_fn = torch.nn.MSELoss().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


def table_to_numpy_double(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.double)

look_back = 4
def train_model(data):
    x_train, y_train= data_split(data, look_back)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    x_train=x_train.to(device)
    y_train=y_train.to(device)
    # Train model
    #####################
    num_epochs = 1000
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
# train the model
learn.learn(
    table = train_dh,
    model_func = train_model,
    inputs = [learn.Input("Price", table_to_numpy_double)],
    outputs = None,
    batch_size = train_dh.size
)

def get_predicted_class(data, idx):
    return data[idx]

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def predict_with_model(data):
    new_data=train_data["Price"].values
    test_data_numpy = np.append(new_data[-99:], data)
    test_data,_=split_sequence(test_data_numpy.reshape(-1,1), 99)
    test_data = torch.from_numpy(test_data).type(torch.Tensor)
    test_data=test_data.to(device)
    y_test_pred = model(test_data)
    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
    y_test_pred=y_test_pred.reshape(1,-1)[0]
    return y_test_pred

# predict on test set
a=learn.learn(
    table = test_dh,
    model_func = predict_with_model,
    inputs = [learn.Input("Price", table_to_numpy_double)],
    outputs = [learn.Output("Predicted_price", get_predicted_class, "double")],
    batch_size = test_dh.size
)

table_pd=dhpd.to_pandas(a)
table_pd['Price'] = scaler.inverse_transform(table_pd['Price'].values.reshape(-1,1))
a=dhpd.to_table(table_pd)
