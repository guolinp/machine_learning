#!/usr/bin/python

import sys
import time
import pandas as pd
import tensorflow as tf


class Sample(object):

    def __init__(self, nr_labels=2):
        self.nr_features = 0
        self.nr_labels = nr_labels

    def load(self, filename):
        samples = pd.DataFrame(pd.read_csv(filename))
        nr_columns = samples.shape[1]  # shape(nr_rows, nr_columns)
        self.nr_features = nr_columns - self.nr_labels
        if self.nr_features <= 0:
            raise Exception("Invalid sample shape, nr_columns={}, nr_features={}, nr_labels={}".format(
                            nr_columns, self.nr_features, self.nr_labels))
        features = samples.ix[:, 0: self.nr_features]
        labels = samples.ix[:, self.nr_features:]
        return self.nr_features, features, labels


class NeuralNetwork(object):

    def __init__(self, nr_labels=1):
        self.nr_hidden_layer_nodes = 4
        self.nr_labels = nr_labels
        self.sample = Sample(self.nr_labels)

    def train(self, filename, learning_rate=1e-5, step=2000, batch=20, target_mse=0.05, verbose=False):
        # load samples
        print("train: start load sample...")
        nr_features, X, Y = self.sample.load(filename)

        # weight
        print("train: prepare train model...")
        self.w1 = tf.Variable(tf.truncated_normal([nr_features, self.nr_hidden_layer_nodes]))
        self.b1 = tf.Variable(tf.random_normal([self.nr_hidden_layer_nodes]))

        self.w2 = tf.Variable(tf.truncated_normal([self.nr_hidden_layer_nodes, self.nr_labels]))
        self.b2 = tf.Variable(tf.random_normal([self.nr_labels]))

        # features
        self.x_features = tf.placeholder(dtype=tf.float32, shape=[None, nr_features])

        # label
        self.y_label = tf.placeholder(dtype=tf.float32, shape=[None, self.nr_labels])

        # self.tf_nn_fn = tf.nn.relu
        self.tf_nn_fn = tf.nn.sigmoid
        print("train: tf_nn_fn: {}".format(self.tf_nn_fn))

        self.t = self.tf_nn_fn(tf.add(tf.matmul(self.x_features, self.w1), self.b1))
        self.y_predict = self.tf_nn_fn(tf.add(tf.matmul(self.t, self.w2), self.b2))

        self.MSE = tf.reduce_mean(abs(self.y_predict - self.y_label))
        self.TRAIN = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.MSE)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print("train: start train...")

        start = 0
        end = batch
        nr_samples = len(X)

        print("train: nr_features: {}, nr_samples: {}, learning rate: {}, step: {}, batch: {}, target_mse: {}".format(
              nr_features, nr_samples, learning_rate, step, batch, target_mse))

        MIN_H = 1e10
        for current_step in range(0, step):
            self.sess.run(self.TRAIN, feed_dict={self.x_features: X[start:end], self.y_label: Y[start:end]})
            if not current_step % 10:
                H = self.sess.run(self.MSE, feed_dict={self.x_features: X[start:end], self.y_label: Y[start:end]})
                # H, w1, w2 = self.sess.run([self.MSE, self.w1, self.w2],
                # feed_dict={self.x_features: X[start:end], self.y_label: Y[start:end]})
                if verbose and not current_step % 200:
                    print("train: step={} mse={}".format(current_step, H))
                    # print w1,w2
                if H < MIN_H:
                    MIN_H = H
                if H < target_mse:
                    break
            start = end if end < nr_samples else 0
            end = start + batch

        print("train: train done, MIN_MSE: {}, CURRENT_MSE: {}, TARGET_MSE: {}, OK? {}".format(
               MIN_H, H, target_mse, "YES" if H <= target_mse else "NO"))

    def predict_batch(self, input_samples):
        result = self.sess.run(self.y_predict, feed_dict={self.x_features: input_samples})
        return result

    def predict_one_sample(self, sample):
        sample_str = ",".join(["{}".format(i) for i in sample])

        sample_dict = {}
        for index, value in enumerate(sample):
            sample_dict["x_{}".format(index)] = [value]

        df = pd.DataFrame(data=sample_dict)
        X = df.ix[:, :]
        result = self.sess.run(self.y_predict, feed_dict={self.x_features: X})
        print("predict_one_sample: input: {}, output: {}".format(sample_str, result[0][0]))

        return result[0][0]


def predict_test(train_sample_filename, test_sample_filename, learning_rate, step, batch, target_mse, train_only=False, nr_labels=2):
    nn = NeuralNetwork(nr_labels=nr_labels)
    nn.train(train_sample_filename, learning_rate=learning_rate, step=step, batch=batch, target_mse=target_mse, verbose=True)

    if train_only:
        return

    print("predict_test: batch test, load datset")
    nr_features, X, Y = nn.sample.load(test_sample_filename)
    X_REAL = X.values
    Y_REAL = Y.values

    if len(Y_REAL) == 0 or len(Y_REAL[0]) != nr_labels:
        print("predict_test: invalid test labels")
        return

    start = time.time()
    Y_PREDICT = nn.predict_batch(X)
    time_used = time.time() - start
    print("predict_test: samples: {}, time used: {:.4f}" .format(len(Y_REAL), time_used))

    if len(Y_PREDICT) == 0 or len(Y_PREDICT[0]) != nr_labels:
        print("predict_test: invalid predict results")
        return

    def F_L(L):  # make a formatted list from L
        return map(lambda x: "{:.6f}".format(x), L)

    def is_predict_success(L1, L2):
        # a simple list compare
        if len(L1) != len(L2):
            return False
        if len(L1) <= 1:
            return False
        for i in range(len(L1) - 1):
            if (L1[i] <= L1[i + 1] and L2[i] > L2[i + 1]) or (L1[i] >= L1[i + 1] and L2[i] < L2[i + 1]):
                return False
        return True

    nr_predict_success = 0
    y_predict = range(nr_labels)
    y_real = range(nr_labels)
    for i in range(len(Y_REAL)):
        x_real = X_REAL[i]
        for j in range(len(Y_REAL[i])):
            y_predict[j] = Y_PREDICT[i][j]
            y_real[j] = Y_REAL[i][j]
        predict_result_str = "FAILED"
        if is_predict_success(F_L(y_predict), F_L(y_real)):
            nr_predict_success += 1
            predict_result_str = "SUCCESS"
        print(">> {:<4d}".format(i))
        print("   features : {}".format(F_L(x_real)))
        print("   labels   : {}".format(F_L(y_real)))
        print("   predict  : {}".format(F_L(y_predict)))
        print("   result   : {}".format(predict_result_str))
    print("\npredict_test: success: {}, total: {}, percentage: {:.4%}".format(
          nr_predict_success, len(Y_REAL), float(nr_predict_success) / len(Y_REAL)))


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "Usage: {} <train_file> <verify_file> <learning_rate> <step> <batch> <target_mse> [train_only]".format(sys.argv[0]))
        print("       e.g:  {} train.csv verify.csv 0.00001 20000 40 0.2".format(sys.argv[0]))
        sys.exit(-1)

    train_filename = sys.argv[1]
    verify_filename = sys.argv[2]
    learning_rate = float(sys.argv[3])
    step = int(sys.argv[4])
    batch = int(sys.argv[5])
    target_mse = float(sys.argv[6])
    train_only = True if len(sys.argv) > 7 else False

    predict_test(train_filename, verify_filename, learning_rate, step, batch, target_mse, train_only=train_only)
