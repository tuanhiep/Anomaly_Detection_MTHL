# @Xian Teng
# @2017

import SVDD
from MTHL import MTHL
import numpy as np
import cPickle
from sklearn.metrics import accuracy_score
import time
import pandas as pd


def normalize(train_data, test_data):
    for v in range(len(train_data)):
        mean = np.mean(train_data[v], axis=0)
        std = np.std(train_data[v], axis=0)
        std[std == 0] = 1
        train_data[v] = (train_data[v] - mean) / std
        test_data[v] = (test_data[v] - mean) / std

    return train_data, test_data


class Config(object):
    def __init__(self):
        self.p = 32
        self.q = 5
        self.lambda1 = 0.12
        self.lambda2 = 1.0
        self.gamma = 1e-3
        self.s = 2
        self.convergence = 1e-8
        self.tolerance = 1e-1


if __name__ == "__main__":
    np.random.seed(2000)
    config = Config()

    # load data
    train = cPickle.load(open("train.pkl", "rb"))
    test = cPickle.load(open("test.pkl", "rb"))
    train_data, train_label = train["data"], train["label"]
    test_data, test_label = test["data"], test["label"]

    # .....normalize datasets....... #
    train_data, test_data = normalize(train_data, test_data)
    # print "loading data finished!"

    # .....run TSHL ...... #
    start_time = time.time()
    tshl = MTHL(train_data, config)
    tshl.Optimization(1e7)
    end_time = time.time()
    print "it costs", float(end_time - start_time), "seconds."

    # predict test data
    train_pred = tshl.predict(train_data, fg="label")
    test_pred = tshl.predict(test_data, fg="label")

    train_acc = accuracy_score(train_label, train_pred, normalize=True)
    test_acc = accuracy_score(test_label, test_pred, normalize=True)
    print "train acc = ", train_acc
    print "test acc = ", test_acc

    # test with cryptojacking_attack data
    nb_views = 3
    dataset_name = "cryptojacking_attack"
    anomaly_rate = 5
    method = "method_2"
    # set up folder for loading data and save the result
    ready_data_directory = "./ready_data"
    # load dataset
    df_views = []
    dim_views = [None] * nb_views
    dataset_path = "{}/{}/anomaly_rate_{}_views_{}/{}".format(ready_data_directory, dataset_name, anomaly_rate,
                                                              nb_views,
                                                              method)
    for i in range(nb_views):
        df_view = pd.read_csv(dataset_path + "/view_%d.csv" % (i + 1),
                              # index_col="Unnamed: 0",
                              # index_col="timestamp",
                              header="infer")
        dim_views[i] = len(df_view.columns)
        df_views.append(df_view)
    # pre-process dataset
    dataset = pd.concat(df_views, axis=1)
