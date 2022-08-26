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
from deephaven import learn
from deephaven.parquet import read
from deephaven import pandas as dhpd
from deephaven import new_table
from deephaven.column import string_col, int_col,double_col

# Python imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# set the device to be GPU
device = 'cuda'
print(device)

# get the latest downloaded data
list_of_files = glob.glob('/mnt/c/Users/yuche/all_data/*') # please put your own file location
latest_file = max(list_of_files, key=os.path.getctime)

# load the data
result = read(latest_file)
data_frame = result.reverse()

# too large, only pick subset of data
data_frame = data_frame.tail_pct(0.02)

# train, test data split 70% train, 30% test
train_dh = data_frame.head_pct(0.7)
test_dh = data_frame.tail_pct(0.3)

# a function to reform the input data
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
look_back = 4

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

# Make the model perform on GPU
model = model.to(device)
loss_fn = torch.nn.MSELoss().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# A function to gather table data into a NumPy ndarray of doubles
def table_to_numpy_double(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.double)

def train_model(data):
    #Scaler the price data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
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
    seq_dim = look_back-1
    for t in range(num_epochs):

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

# Train the model
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


import deephaven.dtypes as dht
from deephaven import DynamicTableWriter
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
coinbase_websocket_table = dtw.table
time_dic={}
def pull_from_coinbase():
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

thread = Thread(target=pull_from_coinbase)
thread.start()


# scatter back data
def get_predicted_class(data, idx):
    return data

First_time = True

def predict_with_model(data):
    global model
    global last_three_data
    global First_time
    data = data[0]
    value = ((data[0]- x_min)/(x_max - x_min))* (1 - (-1)) + (-1)
    if First_time == True:
        First_time = False
        last_three_data = np.array([value,value,value])
        last_three_data = last_three_data.reshape((1, 3, 1))
    else:
        add_data = np.array([value])
        last_three_data = np.append(last_three_data[:,1:,:],[[add_data]],axis=1)
    input_data = torch.from_numpy(last_three_data).type(torch.Tensor).to(device)
    y_test_pred = model(input_data)
    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
    y_test_pred = y_test_pred.reshape(1,-1)[0][0]
    return y_test_pred


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










    

