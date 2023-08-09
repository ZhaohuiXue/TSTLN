import os
import json
import math
import torch
import numpy
import argparse
import weka.core.jvm
import weka.core.converters
import timeit



def load_UEA_dataset(path, dataset):
    """
    Loads the UEA dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # Initialization needed to load a file with Weka wrappers
    weka.core.jvm.start()
    loader = weka.core.converters.Loader(
        classname="weka.core.converters.ArffLoader"
    )

    train_file = os.path.join(path, dataset, dataset + "_TRAIN.arff")
    test_file = os.path.join(path, dataset, dataset + "_TEST.arff")
    train_weka = loader.load_file(train_file)
    test_weka = loader.load_file(test_file)

    train_size = train_weka.num_instances
    test_size = test_weka.num_instances
    nb_dims = train_weka.get_instance(0).get_relational_value(0).num_instances
    length = train_weka.get_instance(0).get_relational_value(0).num_attributes

    train = numpy.empty((train_size, nb_dims, length))
    test = numpy.empty((test_size, nb_dims, length))
    train_labels = numpy.empty(train_size, dtype=numpy.int)
    test_labels = numpy.empty(test_size, dtype=numpy.int)

    for i in range(train_size):
        train_labels[i] = int(train_weka.get_instance(i).get_value(1))
        time_series = train_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            train[i, j] = time_series.get_instance(j).values

    for i in range(test_size):
        test_labels[i] = int(test_weka.get_instance(i).get_value(1))
        time_series = test_weka.get_instance(i).get_relational_value(0)
        for j in range(nb_dims):
            test[i, j] = time_series.get_instance(j).values

    # Normalizing dimensions independently
    for j in range(nb_dims):
        mean = numpy.mean(numpy.concatenate([train[:, j], test[:, j]]))
        var = numpy.var(numpy.concatenate([train[:, j], test[:, j]]))
        train[:, j] = (train[:, j] - mean) / math.sqrt(var)
        test[:, j] = (test[:, j] - mean) / math.sqrt(var)

    # Move the labels to {0, ..., L-1}
    labels = numpy.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train_labels = numpy.vectorize(transform.get)(train_labels)
    test_labels = numpy.vectorize(transform.get)(test_labels)

    weka.core.jvm.stop()
    print('dataset load succeed !!!')

    train_dataset = []
    for i in range(train.shape[0]):
        train_dataset.append((train[i].astype("float32"), int(train_labels[i])))

    test_dataset = []
    for i in range(test.shape[0]):
        test_dataset.append((test[i].astype("float32"), int(test_labels[i])))

    return train_dataset, test_dataset