# package imports
from deephaven.replay import TableReplayer
from deephaven.time import to_datetime

# define replay time range 
start_time = to_datetime("2022-08-09T19:18:00 NY")
end_time = to_datetime("2022-08-11T03:15:00 NY")

replayer = TableReplayer(start_time, end_time)
real_test = replayer.add_table(test_dh, "Date")
replayer.start()

# A function to use the fitted model
def use_fitted_model(features):
    return linear_regression_gpu.predict(features)

# Our scatter function
def scatter(data, idx):
    return data

Predict_table = learn.learn(
    table=real_test,
    model_func=use_fitted_model,
    inputs=[learn.Input(["Price_1","Price_2","Price_3"], table_to_numpy)],
    outputs=[learn.Output("Predicted_Price", scatter, "double")],
    batch_size=test_dh.size
)