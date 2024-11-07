import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
import numpy as np
from models.CNNModel import CNNModel
import argparse
from mia import *
from UnlearningModelRandomLabel import UnlearningModelRandomLabel
import csv
import statistics
#from sklearn.utils import shuffle

def fed_args():
    """
    Arguments for running fedUNRAN
    :return: Arguments for fedUNRAN
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-nc', '--sys-n_client', type=int, required=True, help='Number of the clients')
    parser.add_argument('-nuc', '--sys-n_unlearning_client', type=int, required=True, help='Index of unlearning client')
    parser.add_argument('-ck', '--sys-n_local_class', type=int, required=True, help='Number of the classes in each client')
    parser.add_argument('-ds', '--sys-dataset', type=str, required=True, help='Dataset name, one of the following four datasets: mnist, cifar10, fashion')
    parser.add_argument('-md', '--sys-model', type=str, required=True, help='Model name, only cnn')
    parser.add_argument('-dir', '--sys-dir', type=str, required=True, help='Directory of the results: if result file not exists it will be create')
    parser.add_argument('-lr', '--sys-lr', type=int, required=True, help='Learning rate')
    parser.add_argument('-ne', '--sys-n_epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-nr', '--sys-n_run', type=int, required=True, help='Number of run')

    args = parser.parse_args()
    return args

def create_cnn_model(only_digits=True, seed=1):
    """The CNN model used in https://arxiv.org/abs/1602.05629.
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only EMNIST dataset. If False, uses 62 outputs for the larger
        dataset.
    Returns:
      An uncompiled `tf.keras.Model`.
    """
    print("Create CNN")
    input_shape = [None, 28, 28,1]
    model = CNNModel()
    model.build(input_shape)
    return model


def fedavg(unlearning_model, server_x_test, server_y_test, clients_x_train, clients_y_train, epochs, num_clients, ds_retain, ds_test, 
           ds_forgetting, retrained_test_accuracy = 0, unlearning_test_accuracy = 0, unlearning_client=1, result_file_dir=".", 
           NUM_EXAMPLES_MIA_TUNING=1):
    """
    Method to run fedAVG algorithm on the unlearning model
    :param unlearning_model: unlearning model
    :param server_x_test: x_test dataset
    :param server_y_test: y_test dataset
    :param clients_x_train: x_train dataset of the unlearning client
    :param clients_y_train: y_train dataset of the unlearning client
    :param epochs: clients epoch
    :param num_client: number of clients
    :param ds_retain: dataset create in main function for mia evaluation
    :param ds_test: dataset create in main function for mia evaluation
    :param ds_forgetting: dataset create in main function for mia evaluation
    :param retrained_test_accuracy: test accuracy of the retrained model (evaluate in main function)
    :param unlearning_test_accuracy: test accuracy of the unlearning model (evaluate in main function)
    :param unlearning_client: id of the unlearning client
    :param result_file_dir: dir path of txt file where write results
    :param NUM_EXAMPLES_MIA_TUNING
    :return a dictionary with all results
    """


    print("unlearning_model - Initial evaluate")
    #test_accuracy = unlearning_model.evaluate(server_x_test, server_y_test)
    num_round = 0
    n = 0
    random_client_indexs = list(range(num_clients))
    random_client_indexs.pop(unlearning_client)

    x_train_client_unlearning = clients_x_train[unlearning_client]
    y_train_client_unlearning = clients_y_train[unlearning_client]
    forget_ds = tf.data.Dataset.from_tensor_slices((x_train_client_unlearning, y_train_client_unlearning))
    
    for i in range(num_clients):
        n += len(clients_x_train[i])

    print(f"Client list: {random_client_indexs}")

    if retrained_test_accuracy <= unlearning_test_accuracy:
        mia = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=ds_forgetting, model=unlearning_model)
        results= unlearning_model.evaluate(server_x_test, server_y_test)
        unlearning_test_accuracy = results[1]

        results = unlearning_model.evaluate(forget_ds.batch(32), verbose=2)
        unlearning_forgetting_accuracy = results[1]
    MAX_ROUND = 10
    print("retrain_test_accuracy:", retrained_test_accuracy)
    print("unlearning_test_accuracy:", unlearning_test_accuracy)
    while (retrained_test_accuracy > unlearning_test_accuracy and num_round<=MAX_ROUND):
        print()
        print("Start round ", num_round)
        
        # fedAvg algorithm
        new_global_weights = [np.zeros_like(ww) for ww in unlearning_model.get_weights()]
        for index in random_client_indexs:
            client_x_train = clients_x_train[index]
            client_y_train = clients_y_train[index]

            client_weights_list, num_samples = client_update(unlearning_model, index, client_x_train, client_y_train, num_round, epochs=epochs)
            temp_global_weights = [ww * (num_samples/n) for ww in client_weights_list]

            new_global_weights = [new_global_weights[i] + temp_global_weights[i]  for i in range(len(new_global_weights))]

        unlearning_model.set_weights(new_global_weights)
        
        results= unlearning_model.evaluate(server_x_test, server_y_test)
        unlearning_test_accuracy = results[1]

        results = unlearning_model.evaluate(forget_ds.batch(32), verbose=2)
        unlearning_forgetting_accuracy = results[1]
        
        mia = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=ds_forgetting, model=unlearning_model)
        
        write_results(result_file_dir, f"Unlearned Model (after recovery, recovery round = {num_round})\t accuracy_on_forgetting: {unlearning_forgetting_accuracy} -- accuracy_on_test: {unlearning_test_accuracy} -- mia: {mia}\n")

        print("End round ", num_round)
        num_round += 1
    
    print(f"Per ottenere un'accuracy >= di retrained_accuracy sono necessari {num_round} round")
    write_results(result_file_dir, f"Unlearned Model (after recovery, recovery round needed = {num_round})\t accuracy_on_forgetting: {unlearning_forgetting_accuracy} -- accuracy_on_test: {unlearning_test_accuracy} -- mia: {mia}\n\n")
    
    res = {
        'num_round':num_round,
        'accuracy_on_forgetting':unlearning_forgetting_accuracy,
        'accuracy_on_test':unlearning_test_accuracy,
        'mia':mia,
    }
    return res


def client_update(unlearning_model, client_index, client_x_train, client_y_train, round=0, epochs=1):
    """
    Method to update the client model
    :param server_model:
    :param client_index: index of the current client
    :param clients_x_train: Array off all x_train dataset of all clients
    :param clients_y_train: Array off all y_train datest of all clients
    :param round: Number of the current round
    :param epochs: Number of train epochs
    :return: Client model's weights and the lenght of his dataset
    """
    
    print(f"Update client {client_index} - round: {round}")
    client_model = create_cnn_model()
    client_model.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    client_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    client_model.set_weights(unlearning_model.get_weights())

    length = 0
    client_model.fit(client_x_train, client_y_train, epochs=epochs)
    length = len(client_x_train)

    weights = client_model.get_weights()
    return weights, length


def divide_data(num_client=1, num_local_class=10, dataset_name='emnist', i_seed=0, num_classes=10):
    """
    Method to divide data in each client
    :param num_client: Total number of client
    :param num_local_class: Number of the local classes in each client
    :param dataset_name: 
    :param i_seed: 
    :param num_classes: Total number of classes in the original dataset (dataset_name)
    :return: Dataset of all clients
    """

    if dataset_name=="mnist":
        (train_data, train_targets), (test_data, test_targets) = tf.keras.datasets.mnist.load_data()
    elif dataset_name=="fashion":
        (train_data, train_targets), (test_data, test_targets) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name=="cifar10":
        (train_data, train_targets), (test_data, test_targets) = tf.keras.datasets.cifar10.load_data()
    
    # Data normalization
    train_data = train_data/255.0
    test_data = test_data/255.0
    
    if dataset_name=="cifar10":
        mean = np.mean(train_data, axis=(0,1,2))
        std = np.std(train_data, axis=(0,1,2))
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        train_targets = train_targets.flatten()
        test_targets = test_targets.flatten()
    if not dataset_name=="cifar10":
        train_data = tf.expand_dims(train_data, axis=-1)
        test_data = tf.expand_dims(test_data, axis=-1)
    x_test = test_data
    y_test = test_targets
    print("x_train shape: ", train_data[0].shape)

    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i+j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]

            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    for cls in config_division.keys():
        indexes = tf.where(tf.equal(train_targets, cls))
        num_datapoint = indexes.shape[0]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    for user in (config_class.keys()):
        user_data_indexes = tf.constant([], dtype=tf.int64, shape=[0, 1])
        for cls in config_class[user]:
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = tf.concat([user_data_indexes, user_data_index],axis=0)
            config_data[cls][0] += 1
        user_data_indexes = tf.squeeze(user_data_indexes)
        print(user_data_indexes.shape)
        user_data_indexes = tf.cast(user_data_indexes, dtype=tf.int64)
        user_data_indexes = user_data_indexes.numpy().tolist()

        
        user_data = tf.gather(train_data, user_data_indexes) 
        user_targets = tf.gather(train_targets, user_data_indexes)
        
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = {'x': user_data, 'y': user_targets}
        trainset_config['num_samples'] = len(user_data)

    test_iid_data = {'x': None, 'y': None}
    test_iid_data['x'] = test_data
    test_iid_data['y'] = test_targets

    x_train_clients = []
    y_train_clients = []
    for client_id in trainset_config['users']:
      if dataset_name=="mnist" or  dataset_name=="fashion":
        trainset_config['user_data'][client_id]["x"] = tf.squeeze(trainset_config['user_data'][client_id]["x"])
        trainset_config['user_data'][client_id]["x"] = tf.expand_dims(trainset_config['user_data'][client_id]["x"], axis=-1)
      else:
        print(trainset_config['user_data'][client_id]["x"][1].shape, trainset_config['user_data'][client_id]["y"][1].shape)
        trainset_config['user_data'][client_id]["x"] = tf.squeeze(trainset_config['user_data'][client_id]["x"])
        print(trainset_config['user_data'][client_id]["x"][1].shape, trainset_config['user_data'][client_id]["y"][1].shape)

      x_trainset = tf.convert_to_tensor(trainset_config['user_data'][client_id]["x"])
      y_trainset = tf.convert_to_tensor(trainset_config['user_data'][client_id]["y"])
      x_train_clients.append(x_trainset)
      y_train_clients.append(y_trainset)


    return x_train_clients, y_train_clients, x_test, y_test


def write_results(file_dir, string):
    with open(f"{file_dir}/fedUNRAN_results.txt", "a+") as result_file:
        result_file.write(string)


def evaluate_avg_std(values, unlearning_type, dataset, learning_rate, epoch, run, client_id):
    """
    Method to evaluate all the metrics
    :param values: array of dictionary with results
    :param unlearning_type: name of the unlearning method (FedUNRAN)
    :param dataset: dataset name
    :param learning_rate: learning_rate used during unlearning
    :param epoch: epoch used during unlearning
    :param run: run number
    :param client_id: id of the unlearning client
    :return: a dictionary with all results
    """
   
    accuracy_forgetting_original = []
    accuracy_forgetting_retrained = []
    accuracy_forgetting_unlearning = []
    accuracy_forgetting_after_fedavg = []
    accuracy_test_original = []
    accuracy_test_retrained = []
    accuracy_test_unlearning = []
    accuracy_test_after_fedavg = []
    mia_original = []
    mia_retrained = []
    mia_unleaning = []
    mia_after_fedavg = []
    round = []

    for value in values:
        accuracy_forgetting_original.append(value["accuracy_on_forgetting"]["original"])
        accuracy_forgetting_retrained.append(value["accuracy_on_forgetting"]["retrained"])
        accuracy_forgetting_unlearning.append(value["accuracy_on_forgetting"]["unlearned"])
        accuracy_forgetting_after_fedavg.append(value["accuracy_on_forgetting"]["unlearned_after_fedavg"])

        accuracy_test_original.append(value["accuracy_on_test"]["original"])
        accuracy_test_retrained.append(value["accuracy_on_test"]["retrained"])
        accuracy_test_unlearning.append(value["accuracy_on_test"]["unlearned"])
        accuracy_test_after_fedavg.append(value["accuracy_on_test"]["unlearned_after_fedavg"])

        mia_original.append(value["mia"]["original"])
        mia_retrained.append(value["mia"]["retrained"])
        mia_unleaning.append(value["mia"]["unlearned"])
        mia_after_fedavg.append(value["mia"]["unlearned_after_fedavg"])

        round.append(value["fedavg_round"])


    res =  {
        'unlearning_type': unlearning_type,
        'dataset': dataset,
        'lr': learning_rate,
        'epoch': epoch,
        'num_run': num_run,
        'client_id': client_id,
        'avg':{
            'accuracy_on_forgetting':{
                "original": sum(accuracy_forgetting_original)/run,
                "retrained": sum(accuracy_forgetting_retrained)/run,
                "unlearned": sum(accuracy_forgetting_unlearning)/run,
                "unlearned_after_fedavg": sum(accuracy_forgetting_after_fedavg)/run,
            },
            'accuracy_on_test':{
                "original": sum(accuracy_test_original)/run,
                "retrained": sum(accuracy_test_retrained)/run,
                "unlearned": sum(accuracy_test_unlearning)/run,
                "unlearned_after_fedavg": sum(accuracy_test_after_fedavg)/run,
            },
            'mia':{
                "original": sum(mia_original)/run,
                "retrained": sum(mia_retrained)/run,
                "unlearned": sum(mia_unleaning)/run,
                "unlearned_after_fedavg": sum(mia_after_fedavg)/run,
            },
            "round":sum(round)/run
        },
        'std':{
            'accuracy_on_forgetting':{
                "original": statistics.stdev(accuracy_forgetting_original),
                "retrained": statistics.stdev(accuracy_forgetting_retrained),
                "unlearned": statistics.stdev(accuracy_forgetting_unlearning),
                "unlearned_after_fedavg": statistics.stdev(accuracy_forgetting_after_fedavg),
            },
            'accuracy_on_test':{
                "original": statistics.stdev(accuracy_test_original),
                "retrained": statistics.stdev(accuracy_test_retrained),
                "unlearned": statistics.stdev(accuracy_test_unlearning),
                "unlearned_after_fedavg": statistics.stdev(accuracy_test_after_fedavg),
            },
            'mia':{
                "original": statistics.stdev(mia_original),
                "retrained": statistics.stdev(mia_retrained),
                "unlearned": statistics.stdev(mia_unleaning),
                "unlearned_after_fedavg": statistics.stdev(mia_after_fedavg),
            },
            "round":statistics.stdev(round)
        }
    }
    return res



def flatten_dict(d, parent_key='', sep='.'):
    """
    Utility function used to write results in csv file
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_avg_std_results(result_file_dir, dict):
    """
    Method used to save results in csv file
    :param result_file_dir: dir path
    :param dictionary with all results
    """
    
    write_results(result_file_dir, "\nACCURACY_ON_FORGETTING\n")
    write_results(result_file_dir, f"Original ----> AVG: {dict['avg']['accuracy_on_forgetting']['original']}\tSTD: {dict['std']['accuracy_on_forgetting']['original']}\n")
    write_results(result_file_dir, f"Retrained ----> AVG: {dict['avg']['accuracy_on_forgetting']['retrained']}\tSTD: {dict['std']['accuracy_on_forgetting']['retrained']}\n")
    write_results(result_file_dir, f"Unlearning ----> AVG: {dict['avg']['accuracy_on_forgetting']['unlearned']}\tSTD: {dict['std']['accuracy_on_forgetting']['unlearned']}\n")
    write_results(result_file_dir, f"Unlearning after fed_avg ----> AVG: {dict['avg']['accuracy_on_forgetting']['unlearned_after_fedavg']}\tSTD: {dict['std']['accuracy_on_forgetting']['unlearned_after_fedavg']}\n\n")

    write_results(result_file_dir, "ACCURACY_ON_TEST\n")
    write_results(result_file_dir, f"Original ----> AVG: {dict['avg']['accuracy_on_test']['original']}\tSTD: {dict['std']['accuracy_on_test']['original']}\n")
    write_results(result_file_dir, f"Retrained ----> AVG: {dict['avg']['accuracy_on_test']['retrained']}\tSTD: {dict['std']['accuracy_on_test']['retrained']}\n")
    write_results(result_file_dir, f"Unlearning ----> AVG: {dict['avg']['accuracy_on_test']['unlearned']}\tSTD: {dict['std']['accuracy_on_test']['unlearned']}\n")
    write_results(result_file_dir, f"Unlearning after fed_avg ----> AVG: {dict['avg']['accuracy_on_test']['unlearned_after_fedavg']}\tSTD: {dict['std']['accuracy_on_test']['unlearned_after_fedavg']}\n\n")

    write_results(result_file_dir, "MIA\n")
    write_results(result_file_dir, f"Original ----> AVG: {dict['avg']['mia']['original']}\tSTD: {dict['std']['mia']['original']}\n")
    write_results(result_file_dir, f"Retrained ----> AVG: {dict['avg']['mia']['retrained']}\tSTD: {dict['std']['mia']['retrained']}\n")
    write_results(result_file_dir, f"Unlearning ----> AVG: {dict['avg']['mia']['unlearned']}\tSTD: {dict['std']['mia']['unlearned']}\n")
    write_results(result_file_dir, f"Unlearning after fed_avg ----> AVG: {dict['avg']['mia']['unlearned_after_fedavg']}\tSTD: {dict['std']['mia']['unlearned_after_fedavg']}\n\n")

    write_results(result_file_dir, "RECOVERY ROUND\n")
    write_results(result_file_dir, f"----> AVG: {dict['avg']['round']}\tSTD: {dict['std']['round']}\n\n")

    csv_path = f"{result_file_dir}/fedUNRAN_results.csv"

    data = flatten_dict(dict)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(data.keys())
        
        writer.writerow(data.values())

    return

def main(num_clients, unlearning_client, num_local_classes, 
         dataset_name, result_file_dir, learning_rate, epoch,
         natural_baseline=False, mia_avg=0,):
    """
    Main method
    :param num_clients: number of clients
    :param unlearning_client: id of the unlearning client
    :param num_local_classes: number of classes in each client
    :param dataset_name: name of the dataset
    :param result_file_dir: dir path of txt file where write results
    :param learning_rate: learning_rate of the unlearning model
    :param epoch: number of epoch of the unlearning model
    :return: a dictionary with all results
    """


    write_results(result_file_dir, f"\n\nParameters: Num_clients {num_clients}, num_local_classes {num_local_classes}, \n"\
                      f"unlearning_client {unlearning_client}, learning_rate {learning_rate}, epoch {epoch}, natural_baseline {natural_baseline}, mia_avg {mia_avg}\n")
    
    NUM_EXAMPLES_MIA_TUNING = 10000

    # unlearning phase
    print("Unlearning phase")

    name = f"original_model_{unlearning_client}.keras"
    original_model = tf.keras.models.load_model(name)
    original_model_clone = create_cnn_model()
    original_model_clone.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    original_model_clone.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    original_model_clone.set_weights(original_model.get_weights())

    
    x_train, y_train, x_test, y_test = divide_data(num_client=num_clients, num_local_class=num_local_classes, dataset_name=dataset_name, i_seed=0, num_classes=10)
    z_train_client_unlearning = np.array([])
    
    # load unlearning client's (client UNLEARNING_CLIENT) data
    x_train_client_unlearning = x_train[unlearning_client]
    y_train_client_unlearning = y_train[unlearning_client]
    z_train_client_unlearning = np.full(y_train_client_unlearning.shape, 1)
    
    ds_train_client_unlearning = tf.data.Dataset.from_tensor_slices((x_train_client_unlearning, y_train_client_unlearning, z_train_client_unlearning))

    # pre-processing
    ds_train_client_unlearning = ds_train_client_unlearning.map(lambda x, y, z: (x, y, tf.cast(z, tf.int32)))
    ds_train_client_unlearning = ds_train_client_unlearning.shuffle(1032).batch(32)
    
    unlearning_model = UnlearningModelRandomLabel(original_model_clone)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True,)
    
    unlearning_model.compile(optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

    if not natural_baseline:
        unlearning_model.fit(ds_train_client_unlearning, epochs=epoch, callbacks=[callback])
    
    
    x_train_client_unlearning = x_train[unlearning_client]
    y_train_client_unlearning = y_train[unlearning_client]
    forget_ds = tf.data.Dataset.from_tensor_slices((x_train_client_unlearning, y_train_client_unlearning,))

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    indexes = list(range(num_clients))
    indexes.pop(unlearning_client)

    ds_retain = tf.data.Dataset.from_tensor_slices(
        (np.vstack([x_train[i] for i in indexes]),
        np.concatenate([y_train[i] for i in indexes]))
    )

    ds_retain = ds_retain.shuffle(54000, reshuffle_each_iteration=False)


    forget_ds = forget_ds.shuffle(1032).batch(32)
    
    ds_test = ds_test.shuffle(1032*10, reshuffle_each_iteration=False)

    # evaluation - original model
    results = original_model.evaluate(forget_ds, verbose=2)
    accuracy_on_forgetting_original = results[1]
    results = original_model.evaluate(ds_test.batch(32), verbose=2)
    accuracy_on_test_original = results[1]
    mia_original = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=forget_ds, model=original_model)

    print("Original Model")
    print(f"accuracy_on_forgetting: {accuracy_on_forgetting_original} -- accuracy_on_test: {accuracy_on_test_original} -- mia: {mia_original}")
    write_results(".",f"Original Model\t accuracy_on_forgetting: {accuracy_on_forgetting_original} -- accuracy_on_test: {accuracy_on_test_original} -- mia: {mia_original}\n")

    # evaluation - retrained model
    name = f"retrained_model_{unlearning_client}.keras"
    retrained_model = tf.keras.models.load_model(name)
    results = retrained_model.evaluate(forget_ds, verbose=2)
    accuracy_on_forgetting_retrained = results[1]
    results = retrained_model.evaluate(ds_test.batch(32), verbose=2)
    accuracy_on_test_retrained = results[1]
    mia_retrained = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=forget_ds, model=retrained_model)
    print("Retrained Model")
    print(f"accuracy_on_forgetting: {accuracy_on_forgetting_retrained} -- accuracy_on_test: {accuracy_on_test_retrained} -- mia: {mia_retrained}")
    write_results(".",f"Retrained Model\t accuracy_on_forgetting: {accuracy_on_forgetting_retrained} -- accuracy_on_test: {accuracy_on_test_retrained} -- mia: {mia_retrained}\n")

    # evaluation - unlearning model
    unlearning_model_to_evaluate = create_cnn_model()
    unlearning_model_to_evaluate.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    unlearning_model_to_evaluate.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    unlearning_model_to_evaluate.set_weights(unlearning_model.get_weights())
    results = unlearning_model_to_evaluate.evaluate(forget_ds, verbose=2)

    accuracy_on_forgetting_unlearning = results[1]
    results =  unlearning_model_to_evaluate.evaluate(ds_test.batch(32), verbose=2)
    accuracy_on_test_unlearning = results[1]
    mia_unlearning = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=forget_ds, model=unlearning_model_to_evaluate)
    print("Unlearned Model")
    print(f"accuracy_on_forgetting: {accuracy_on_forgetting_unlearning} -- accuracy_on_test: {accuracy_on_test_unlearning} -- mia: {mia_unlearning}")
    write_results(".",f"Unlearned Model\t accuracy_on_forgetting: {accuracy_on_forgetting_unlearning} -- accuracy_on_test: {accuracy_on_test_unlearning} -- mia: {mia_unlearning}\n")

    fedavg_epochs = 1
    fedavg_res = fedavg(unlearning_model_to_evaluate, 
        x_test, 
        y_test, 
        x_train, 
        y_train,
        fedavg_epochs,
        num_clients,
        ds_retain,
        ds_test,
        forget_ds, 
        accuracy_on_test_retrained, 
        accuracy_on_test_unlearning,
        unlearning_client,
        ".",
        NUM_EXAMPLES_MIA_TUNING
        )
    res = {
        'accuracy_on_forgetting':{
            'original':accuracy_on_forgetting_original,
            'retrained':accuracy_on_forgetting_retrained,
            'unlearned':accuracy_on_forgetting_unlearning,
            'unlearned_after_fedavg':fedavg_res['accuracy_on_forgetting']
        },
        'accuracy_on_test':{
            'original':accuracy_on_test_original,
            'retrained':accuracy_on_test_retrained,
            'unlearned':accuracy_on_test_unlearning,
            'unlearned_after_fedavg':fedavg_res['accuracy_on_test']
        },
        'mia':{
            'original':mia_original,
            'retrained':mia_retrained,
            'unlearned':mia_unlearning,
            'unlearned_after_fedavg':fedavg_res['mia']
        },
        'fedavg_round':fedavg_res['num_round']
    }

    return res



if __name__ == "__main__":

    args = fed_args()

    # TOGLIERE only_unlearning_client (è True), IL CICLO SUI 5 CLIENT MA PASSARE CLIENT_UNLEARNING_ID COME ARGOMENTO

    num_clients = args.sys_n_client
    unlearning_client = args.sys_n_unlearning_client
    num_local_classes = args.sys_n_local_class #10: IID, 2: NON-IID
    dataset_name = args.sys_dataset
    model_name = args.sys_model
    result_file_dir = args.sys_dir
    learning_rate = args.sys_lr
    num_epochs = args.sys_n_epochs
    num_run = args.sys_n_run


    avaible_datasets = ["mnist", "fashion", "cifar10"]
    if dataset_name not in avaible_datasets:
        raise Exception(f"Dataset not avaible. Select one between of {avaible_datasets}")
    
    avaible_models = ["cnn"]
    if model_name not in avaible_models:
        raise Exception(f"Model not avaible. Select one between of {avaible_models}")
    
    run_results = []
    
    for i in range(3):
        res = main(num_clients,
            unlearning_client,
            num_local_classes,
            dataset_name,
            result_file_dir,
            learning_rate,
            num_epochs)
        run_results.append(res)
    res = evaluate_avg_std(run_results, "fedUNRAN", dataset_name, learning_rate, num_epochs, num_run, unlearning_client)
    save_avg_std_results(result_file_dir, res)