#!/usr/bin/python

# Reference:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python

import cPickle
import argparse
from random import seed
from random import randrange
from csv import reader


# Store tree model
def store_tree(tree, filename=".tree"):
    cPickle.dump(tree, open(filename, "wb"))


# Grab tree model
def grab_tree(filename=".tree"):
    return cPickle.load(open(filename))


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rb")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[
                    index], gini, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del(node["groups"])
    # check for a no split
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split(node["left"], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split(node["right"], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    store_tree(tree)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


# Evaluate an algorithm using a cross validation split
def evaluate(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Evaluate algorithm
def evaluate_algorithm(dataset, n_folds, max_depth, min_size):
    # Test CART on Bank Note dataset
    seed(1)
    # evaluate algorithm
    scores = evaluate(dataset, decision_tree, n_folds, max_depth, min_size)
    print("scores: {}".format(scores))
    print("mean accuracy: {:.4}".format((sum(scores) / float(len(scores)))))


# Decision tree prediction
def predict_run(predict_dataset):
    nr_predict_success = 0
    nr_total_dataset = len(predict_dataset)

    tree = grab_tree()

    predictions = list()
    for row in predict_dataset:
        prediction = predict(tree, row)
        predictions.append(prediction)

    for i,value in enumerate(predict_dataset):
        if value[-1] == predictions[i]:
            result = "SUCCESS"
            nr_predict_success += 1
        else:
            result = " FAILED"
        print("{} : {} - {}".format(result, predictions[i], value[-1]))

    percentage = float(nr_predict_success) / nr_total_dataset
    print("success/total: {}/{}, {:.2%}".format(nr_predict_success, nr_total_dataset, percentage))


# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Decision tree", add_help=True)
    parser.add_argument('-d', '--dataset', help='dataset file', default="data.csv")
    parser.add_argument('-e', '--evaluate', help='evaluate algorithm', type=int, default=1)
    parser.add_argument('-p', '--predict', help='predict sample', type=int, default=0)
    # We will ignore the unknown args.
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # print args

    print("load train dataset from file: {}".format(args.dataset))
    dataset = load_csv(args.dataset)

    # Convert string attributes to floats
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    if args.evaluate != 0:
        print("start evaluate algorithm")
        evaluate_algorithm(dataset, 2, 10, 5)

    if args.predict != 0:
        print("start predict")
        predict_run(dataset)
