# @Xian Teng
# @2017
import argparse
import os

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import SVDD
from MTHL import MTHL
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import time
import pandas as pd


def prepare_directory(log_path):
    done_runtime = log_path + "/done_runtime.txt"
    if os.path.exists(done_runtime):
        return True
    else:
        folder = log_path
        if not os.path.exists(folder):
            os.makedirs(folder)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        return False


def create_train_data(df_view, nb_time_steps):
    my_list = []
    nb_rows = df_view.shape[0]
    for index in range(0, nb_rows, nb_time_steps):
        my_list.append(df_view[index:index + nb_time_steps].transpose().to_numpy())
    mat = np.array(my_list)
    return mat


def normalize(train_data, test_data):
    for v in range(len(train_data)):
        mean = np.mean(train_data[v], axis=0)
        std = np.std(train_data[v], axis=0)
        std[std == 0] = 1
        train_data[v] = (train_data[v] - mean) / std
        test_data[v] = (test_data[v] - mean) / std

    return train_data, test_data


class Config(object):
    def __init__(self, latent_dim):
        self.p = latent_dim
        # self.p = 32
        self.q = 5
        # self.q = 10
        self.lambda1 = 0.12
        self.lambda2 = 1.0
        self.gamma = 1e-3
        self.s = 2
        self.convergence = 1e-8
        self.tolerance = 1e-1


if __name__ == "__main__":
    np.random.seed(2000)

    # load data: this following commented out code of Teng was written in python 2.x
    # train = pickle.load(open("train_unix.pkl", "rb"))
    # test = pickle.load(open("test_unix.pkl", "rb"))
    # train_data, train_label = train["data"], train["label"]
    # test_data, test_label = test["data"], test["label"]
    #
    # # .....normalize datasets....... #
    # train_data, test_data = normalize(train_data, test_data)
    # # print "loading data finished!"
    #
    # # .....run TSHL ...... #
    # start_time = time.time()
    # tshl = MTHL(train_data, config)
    # tshl.Optimization(1e7)
    # end_time = time.time()
    # print(f"it costs {float(end_time - start_time)} seconds.")
    #
    # # predict test data
    # train_pred = tshl.predict(train_data, fg="label")
    # test_pred = tshl.predict(test_data, fg="label")
    #
    # train_acc = accuracy_score(train_label, train_pred, normalize=True)
    # test_acc = accuracy_score(test_label, test_pred, normalize=True)
    # print(f"train acc = {train_acc}")
    # print(f"test acc = {test_acc}")

    # Test with cryptojacking_attack data

    # To run the program from command line:
    # -test_name vary_K - dataset cryptojacking_attack - method method_2 - nb_views 3 - nb_time_steps 10 - anomaly_rate
    # 5 - nb_instances 20 - latent_dim 8 - index_test 2

    # read arguments from command line
    parser = argparse.ArgumentParser(description="State Space Model - Variational AutoEncoder")
    parser.add_argument("-dataset", "--dataset", type=str, required=True, help="the name of input data set")
    parser.add_argument("-test_name", "--test_name", type=str, required=True, help="the name running machine")
    parser.add_argument("-method", "--method", type=str, required=True, help="the method of generated input dataset")
    parser.add_argument("-nb_views", "--nb_views", type=int, required=True, help="the number of views")
    parser.add_argument("-nb_time_steps", "--nb_time_steps", type=int, required=True,
                        help="the number of time steps, Ex: 10")
    parser.add_argument("-anomaly_rate", "--anomaly_rate", type=int, required=True,
                        help="the anomaly rate in %, Ex: 5")
    parser.add_argument("-nb_instances", "--nb_instances", type=int, required=True,
                        help="the number of instances in dataset, Ex: 8084")
    parser.add_argument("-latent_dim", "--latent_dim", type=int, required=True,
                        help="the dimension of latent space, Ex: 5")
    parser.add_argument("-index_test", "--index_test", type=int, required=True, help="the index of current test")
    args = parser.parse_args()

    # Prepare a dataset
    dataset_name = args.dataset
    test_name = args.test_name
    method = args.method
    nb_views = args.nb_views
    nb_time_steps = args.nb_time_steps
    anomaly_rate = args.anomaly_rate
    nb_instances = args.nb_instances
    latent_dim = args.latent_dim
    index_test = args.index_test

    # set up folder for loading data and save the result
    ready_data_directory = "./ready_data"
    result_directory = "./MTHL"

    log_path = f"{result_directory}/{test_name}/{dataset_name}/" + "K_dim_" + str(latent_dim) + \
               "_T_" + str(nb_time_steps) + "_" + method + \
               "_nb_instances_" + str(nb_instances) + \
               "_anomaly_rate_" + str(anomaly_rate) \
               + "_test_" + str(index_test)
    check_done = prepare_directory(log_path)
    if check_done is True:
        print("already done for setting " + log_path)
    else:
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
            train_data_view_i = create_train_data(normalized_df_view, nb_time_steps)
            df_views.append(train_data_view_i)
        # ground truth
        ground_truth = pd.read_csv(dataset_path + "/ground_truth.csv", header="infer")

        # train_data is of dimension: nb_views x nb_instances x nb_features x nb_time_step
        train_data = df_views
        train_label = ground_truth["is_anomaly"].head(nb_instances)

        # .....run TSHL ...... #
        config = Config(latent_dim=latent_dim)
        start_time = time.time()
        tshl = MTHL(train_data, config)
        tshl.Optimization(1e7)
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        print(f"it costs {training_time} minutes.")
        print(f"Finish for setting {log_path}")
        with open(f"{log_path}/done_runtime.txt", "w") as f:
            f.write(f"{training_time} \n")
        # predict test data
        train_score = tshl.predict(train_data, fg="score")
        # score= np.prod(train_score,axis=0)
        score = np.sum(train_score, axis=0)
        auc = metrics.roc_auc_score(train_label, score)
        print(f"auc = {auc}")

        # Save tshl
        tshl_file = open(log_path + f"/tshl.pkl", 'ab')
        pickle.dump(tshl, tshl_file)
        tshl_file.close()
        # Save train_score
        train_score_file = open(log_path + f"/train_score.pkl", 'ab')
        pickle.dump(train_score, train_score_file)
        train_score_file.close()

        # predict test data
        # train_pred = tshl.predict(train_data, fg="label")
        # train_label = ground_truth["is_anomaly"].apply(lambda x: 1-x).head(nb_instances)
        # train_acc = accuracy_score(train_label, train_pred, normalize=True)
        # print "train acc = ", train_acc
