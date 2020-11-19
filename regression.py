import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
        df = pandas.read_csv("https://storage.googleapis.com/kubric-hiring/linreg_train.csv", header= None,index_col=0)
    df_test = pandas.read_csv("https://storage.googleapis.com/kubric-hiring/linreg_test.csv", header= None, index_col=0)
    X_test = numpy.array(df.iloc[0]).reshape(-1,1)
    X_train = numpy.array(df_test.iloc[0]).reshape(-1,1)
    Y_test = numpy.array(df.iloc[1]).reshape(-1,1)
    Y_train = numpy.array(df.iloc[1]).reshape(-1,1)


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
