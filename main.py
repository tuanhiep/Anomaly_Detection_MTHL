# @Xian Teng
# @2017
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

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
        self.p = 8
        # self.p = 32
        self.q = 5
        self.lambda1 = 0.12
        self.lambda2 = 1.0
        self.gamma = 1e-3
        self.s = 2
        self.convergence = 1e-8
        self.tolerance = 1e-1


def create_train_data(df_view, nb_time_steps):
    my_list = []
    nb_rows = df_view.shape[0]
    for index in range(0, nb_rows, nb_time_steps):
        my_list.append(df_view[index:index + nb_time_steps].transpose().to_numpy())
    mat = np.array(my_list)
    return mat


if __name__ == "__main__":
    np.random.seed(2000)
    config = Config()

    # load data
    train = cPickle.load(open("train_unix.pkl", "rb"))
    test = cPickle.load(open("test_unix.pkl", "rb"))
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
    nb_instances = 8085
    nb_time_steps = 10
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
    # normalize dataset
    scaler = StandardScaler()
    for i in range(nb_views):
        df_view = pd.read_csv(dataset_path + "/view_%d.csv" % (i + 1),
                              # index_col="Unnamed: 0",
                              # index_col="timestamp",
                              header="infer")
        dim_views[i] = len(df_view.columns)
        normalized_df_view_numpy = scaler.fit_transform(df_view.head(nb_time_steps * nb_instances))
        normalized_df_view = pd.DataFrame(normalized_df_view_numpy)
        # if i==2:
        #     fish_frame = normalized_df_view.iloc[:, :-1]
        #     train_data_view_i = create_train_data(fish_frame, nb_time_steps)
        # else:
        #     train_data_view_i = create_train_data(normalized_df_view, nb_time_steps)
        train_data_view_i = create_train_data(normalized_df_view, nb_time_steps)
        df_views.append(train_data_view_i)
    # ground truth
    ground_truth = pd.read_csv(dataset_path + "/ground_truth.csv", header="infer")

    # train_data is of dimension: nb_views x nb_instances x nb_features x nb_time_step
    train_data = df_views
    train_label = ground_truth["is_anomaly"].head(nb_instances)

    # .....run TSHL ...... #
    start_time = time.time()
    tshl = MTHL(train_data, config)
    tshl.Optimization(1e7)
    end_time = time.time()
    print "it costs", float(end_time - start_time), "seconds."

    # # predict test data
    # train_score = tshl.predict(train_data, fg="score")
    # auc = metrics.roc_auc_score(train_label, train_score)
    # print "auc = ", auc

    # predict test data
    train_pred = tshl.predict(train_data, fg="label")

    train_acc = accuracy_score(train_label, train_pred, normalize=True)
    print "train acc = ", train_acc

