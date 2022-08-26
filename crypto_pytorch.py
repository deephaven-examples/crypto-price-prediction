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
