from deephaven_server import Server
s = Server(port=10000, jvm_args=["-Xmx4g"])
s.start()
# UGP lock is automatically acquired for each query operation
from deephaven import ugp
ugp.auto_locking = True
import glob
import os

# Deephaven imports
from deephaven import dtypes as dht
from deephaven.learn import gather
from deephaven import learn
from deephaven.parquet import read
from deephaven import pandas as dhpd
from deephaven import new_table
from deephaven.column import string_col, int_col,double_col

# Python imports
from sklearn.preprocessing import MinMaxScaler

import numpy as np

list_of_files = glob.glob('/mnt/c/Users/yuche/all_data/*') # please put your own file location
latest_file = max(list_of_files, key=os.path.getctime)
result = read(latest_file)
data_frame = dhpd.to_pandas(result)


import cudf; print('cuDF Version:', cudf.__version__)
data_frame = cudf.from_pandas(data_frame)
data_frame=data_frame.iloc[::-1]
# too large, only pick subset of data
data_frame["Price_1"]=data_frame["Price"].shift(1)
data_frame["Price_2"]=data_frame["Price"].shift(2)
data_frame["Price_3"]=data_frame["Price"].shift(3)
data_size=int(len(data_frame)*0.98)
data_frame=data_frame.iloc[data_size:]
data_frame=data_frame.reset_index(drop=True)
data_frame.dropna(inplace=True)
data_frame=data_frame.reset_index(drop=True)
train_size=int(len(data_frame)*0.7)
train_data=data_frame.iloc[:train_size]
test_data=data_frame.iloc[train_size:]
train_data=train_data.to_pandas()
test_data=test_data.to_pandas()
train_dh=dhpd.to_table(train_data)
test_dh=dhpd.to_table(test_data)


import cuml
from cuml.linear_model import LinearRegression as LinearRegression_GPU







# Create a linear regression model
linear_regression_gpu = LinearRegression_GPU()

# A function to fit our linear model
def fit_linear_model(features, target):
    linear_regression_gpu.fit(features, target)

# A function to use the fitted model
def use_fitted_model(features):
    return linear_regression_gpu.predict(features)

# Our gather function
def table_to_numpy(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.double)

# Our scatter function
def scatter(data, idx):
    return data[idx]

# Train the linear regression model
learn.learn(
    table=train_dh,
    model_func=fit_linear_model,
    inputs=[learn.Input(["Price_1","Price_2","Price_3"], table_to_numpy), learn.Input("Price", table_to_numpy)],
    outputs=None,
    batch_size=train_dh.size
)

a = learn.learn(
    table=test_dh,
    model_func=use_fitted_model,
    inputs=[learn.Input(["Price_1","Price_2","Price_3"], table_to_numpy)],
    outputs=[learn.Output("Predicted_Price", scatter, "double")],
    batch_size=test_dh.size
)

from deephaven.replay import TableReplayer
from deephaven.time import to_datetime

start_time = to_datetime("2022-07-27T21:10:00 NY")
end_time = to_datetime("2022-07-29T03:15:00 NY")

replayer = TableReplayer(start_time, end_time)
replayed_table = replayer.add_table(a, "Date")
replayer.start()

