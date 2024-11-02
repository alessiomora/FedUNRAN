import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
import random as rd
import numpy as np
from models.CNNModel import CNNModel
import argparse
#from UnlearningModel import UnlearningModel
from sklearn.mixture import GaussianMixture
from mia import *
#from UnlearningModelDistilledSamples import UnlearningModelDistilledSamples
from UnlearningModelRandomLabel import UnlearningModelRandomLabel
#from ModelKLDivAdaptive import ModelKLDivAdaptive

import csv

import statistics
from sklearn.utils import shuffle

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fed_args():
    """
    Arguments for running FedD3
    :return: Arguments for FedD3
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-nc', '--sys-n_client', type=int, required=True, help='Number of the clients')
    parser.add_argument('-nuc', '--sys-n_unlearning_client', type=int, required=True, help='Index of unlearning client')
    parser.add_argument('-uoc', '--sys-unlearning_only_one_client', type=str2bool, required=True, help='Unlearning phase made with only unlearning client')
    parser.add_argument('-ck', '--sys-n_local_class', type=int, required=True, help='Number of the classes in each client')
    #parser.add_argument('-cnd', '--sys-client-n_dd', type=int, required=True, help='Number of distilled images in each client')
    parser.add_argument('-ds', '--sys-dataset', type=str, required=True, help='Dataset name, one of the following four datasets: MNIST, CIFAR-10, fashion, SVHN')
    parser.add_argument('-md', '--sys-model', type=str, required=True, help='Model name')
    parser.add_argument('-dir', '--sys-dir', type=str, required=True, help='Directory of the results: if result file not exists it will be create')

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


def fedavg(unlearning_model, server_x_test, server_y_test, clients_x_train, clients_y_train, epochs, num_clients, ds_retain, ds_test, ds_forgetting, 
           retrained_test_accuracy = 0, unlearning_test_accuracy = 0, unlearning_client=1, result_file_dir=".", 
           NUM_EXAMPLES_MIA_TUNING=1, natural_baseline=False,mia_avg=0):
    
    print("unlearning_model - Initial evaluate")
    #test_accuracy = unlearning_model.evaluate(server_x_test, server_y_test)
    num_round = 0
    n = 0
    mia = float('inf')
    mia_prev = None
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
        #test_loss, unlearning_forgetting_accuracy, sparse_categorical_crossentropy = unlearning_model.evaluate(ds_forgetting.batch(32))
        unlearning_forgetting_accuracy = results[1]
    MAX_ROUND = 10
    print(retrained_test_accuracy)
    print(unlearning_test_accuracy)
    while (retrained_test_accuracy > unlearning_test_accuracy and num_round<=MAX_ROUND):
        print()
        print("Start round ", num_round)
        #new_global_weights = np.zeros_like(unlearning_model.get_weights(), dtype=object)
        """
        random_client_indexs = []
        n = 0

        random_client_indexs = range(1, num_clients) # Tutti client tranne il primo (0)
        for i in range(1, num_clients):
            n += len(x_train_clients[i])
        print("Select ", len(random_client_indexs), " clients")
        """
        # fedAvg algorithm
        new_global_weights = [np.zeros_like(ww) for ww in unlearning_model.get_weights()]
        for index in random_client_indexs:
            client_x_train = clients_x_train[index]
            client_y_train = clients_y_train[index]

            client_weights_list, num_samples = client_update(unlearning_model, index, client_x_train, client_y_train, num_round, epochs=epochs)
            temp_global_weights = [ww * (num_samples/n) for ww in client_weights_list]

            new_global_weights = [new_global_weights[i] + temp_global_weights[i]  for i in range(len(new_global_weights))]
            """
            temp_global_weights = np.zeros_like(unlearning_model.get_weights(), dtype=object)
            client_w, num_samples = client_update(unlearning_model.get_weights(), index, num_round)
            temp_global_weights = client_w
            temp_global_weights = [weight * (num_samples/n) for weight in temp_global_weights]
            new_global_weights += temp_global_weights
            """

        unlearning_model.set_weights(new_global_weights)
        
        results= unlearning_model.evaluate(server_x_test, server_y_test)
        unlearning_test_accuracy = results[1]

        results = unlearning_model.evaluate(forget_ds.batch(32), verbose=2)
        #test_loss, unlearning_forgetting_accuracy, sparse_categorical_crossentropy = unlearning_model.evaluate(ds_forgetting.batch(32))
        unlearning_forgetting_accuracy = results[1]
        
        mia = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=ds_forgetting, model=unlearning_model)
        
        write_results(result_file_dir, f"Unlearned Model (after recovery, recovery round = {num_round})\t accuracy_on_forgetting: {unlearning_forgetting_accuracy} -- accuracy_on_test: {unlearning_test_accuracy} -- mia: {mia}\n")
        #write_results(f"\nRound number {num_round} ------> new MIA unlearning_model results: {mia_unlearning_updated} \t unlearning_test_accuracy: {unlearning_test_accuracy}")
        #write_results(True, False, num_round, mia_unlearning_updated, unlearning_test_accuracy)

        print("End round ", num_round)
        num_round += 1
        if natural_baseline and not num_round==1: # se è il caso natural_baseline e non è il primo round
            if mia_prev<=mia_avg and not mia>=mia_prev+0.008: # se il valore di mia precedente è <= alla media e quello attuale è vicino al valore precedente allora è okay e mi fermo
                break
        mia_prev = mia
    
    print(f"Per ottenere un'accuracy >= di retrained_accuracy sono necessari {num_round} round")
    write_results(result_file_dir, f"Unlearned Model (after recovery, recovery round needed = {num_round})\t accuracy_on_forgetting: {unlearning_forgetting_accuracy} -- accuracy_on_test: {unlearning_test_accuracy} -- mia: {mia}\n\n")
    #write_results("\nfedAVG final results:")
    #write_results(f"\nTo achieve an unlearning_model_test_accuracy >= to the retrained_model_test_accuracy, {num_round} rounds are necessary")
    #write_results(f"\nFinal unlearning_model MIA results after fedAVG: {mia_unlearning_updated}\n\n")
    #write_results(True, True, num_round, mia_unlearning_updated)
    res = {
        'num_round':num_round,
        'accuracy_on_forgetting':unlearning_forgetting_accuracy,
        'accuracy_on_test':unlearning_test_accuracy,
        'mia':mia,
    }
    return res #num_round, unlearning_model


def client_update(unlearning_model, client_index, client_x_train, client_y_train, round=0, epochs=1):
    
    print(f"Update client {client_index} - round: {round}")
    client_model = create_cnn_model()
    client_model.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # qui occhio a usare sparse_categorical_accuracy perchè non prederebbe from_logits=True
    client_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    client_model.set_weights(unlearning_model.get_weights())

    length = 0
    client_model.fit(client_x_train, client_y_train, epochs=epochs)
    length = len(client_x_train)

    weights = client_model.get_weights()
    return weights, length


def client_gmm_coreset(x_train, y_train, client_id, k, num_local_class, i_seed=0):
    """
    The client run the FedD3 with coreset-based instance.
    :param x_train: x_train dataset of the client
    :param y_train: y_train dataset of the client
    :param client_id: Index of the client
    :param k: number of the local distilled images, need to be integral times of number of local classes
    :param num_local_class: Number of the local classes
    :return: Distilled images from decentralized dataset in this client
    """

    print("\nClient: ", client_id)
    print(k)
    res = []
    print(x_train[0].shape)
    x_train_n = tf.squeeze(x_train)
    y_train_n = tf.squeeze(y_train)
    print(x_train[0].shape)
    #num_datapoint = int(k / num_local_class)
    cls_set = set()
    for cls in y_train:
        cls_set.add(cls.item())

    for cls in cls_set:
        sub_data = []
        indexes = tf.where(tf.equal(y_train_n, cls)) #torch.nonzero(self._train_data['y'] == cls)
        indexes = tf.random.shuffle(indexes) #indexes = indexes[torch.randperm(indexes.shape[0])]
        for index in indexes:
            sub_data.append(x_train_n[index.numpy()[0]].numpy().reshape(-1).tolist()) #self._train_data['x'][index].numpy().reshape(-1).tolist())
        sub_data = np.array(sub_data)
        #gm = GaussianMixture(n_components=int(len(cls_set) / len(cls_set)), random_state=0).fit(sub_data)
        gm = GaussianMixture(n_components=int(k / num_local_class), random_state=0).fit(sub_data)
        for x_data in gm.means_:
            k_data_point = [cls, np.array(x_data).reshape(28, 28, 1), k]#(32, 32, 3), k]#(28, 28, 1), len(cls_set)]
            res.append(k_data_point)

    return res

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

        train_targets = train_targets.flatten() #reshape(-1,)#  #bidimensional array -> unidimensional array
        test_targets = test_targets.flatten() #reshape(-1,)#
    if not dataset_name=="cifar10":
        # (28, 28) -> (28, 28, 1)
        train_data = tf.expand_dims(train_data, axis=-1)
        test_data = tf.expand_dims(test_data, axis=-1)
    x_test = test_data
    y_test = test_targets
    #server_x_train = train_data
    #server_y_train = train_targets
    print("x_train shape: ", train_data[0].shape)

    #num_classes, train_data, train_targets, test_data, test_targets = load_data()

    # import pdb; pdb.set_trace()
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
        indexes = tf.where(tf.equal(train_targets, cls)) #torch.nonzero(train_targets == cls)
        num_datapoint = indexes.shape[0]
        #indexes = tf.random.shuffle(indexes) #indexes[torch.randperm(num_datapoint)]
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
            user_data_indexes = tf.concat([user_data_indexes, user_data_index],axis=0) #torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1
        #user_data_indexes = user_data_indexes.squeeze().int().tolist()
        user_data_indexes = tf.squeeze(user_data_indexes)
        print(user_data_indexes.shape)
        user_data_indexes = tf.cast(user_data_indexes, dtype=tf.int64)
        user_data_indexes = user_data_indexes.numpy().tolist()

        
        user_data = tf.gather(train_data, user_data_indexes) #user_data = Subset(trainset, user_data_indexes)
        user_targets = tf.gather(train_targets, user_data_indexes) #user_targets = trainset.target[user_data_indexes.tolist()]
        
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = {'x': user_data, 'y': user_targets}
        trainset_config['num_samples'] = len(user_data)

    test_iid_data = {'x': None, 'y': None}
    test_iid_data['x'] = test_data
    test_iid_data['y'] = test_targets

    x_train_clients = []
    y_train_clients = []
    for client_id in trainset_config['users']:
      # print("Update dimensions")
      if dataset_name=="mnist" or  dataset_name=="fashion":
        #shape (1,28,28,1) -> (28,28,1)
        trainset_config['user_data'][client_id]["x"] = tf.squeeze(trainset_config['user_data'][client_id]["x"])
        trainset_config['user_data'][client_id]["x"] = tf.expand_dims(trainset_config['user_data'][client_id]["x"], axis=-1)
      else:
        #shape (1,32,32,3) -> (32,32,3)
        print(trainset_config['user_data'][client_id]["x"][1].shape, trainset_config['user_data'][client_id]["y"][1].shape)
        trainset_config['user_data'][client_id]["x"] = tf.squeeze(trainset_config['user_data'][client_id]["x"])
        print(trainset_config['user_data'][client_id]["x"][1].shape, trainset_config['user_data'][client_id]["y"][1].shape)
        #trainset_config['user_data'][client_id]["x"] = [tf.reshape(tensor, (32, 32, 3)) for tensor in trainset_config['user_data'][client_id]["x"]]

      x_trainset = tf.convert_to_tensor(trainset_config['user_data'][client_id]["x"])
      y_trainset = tf.convert_to_tensor(trainset_config['user_data'][client_id]["y"])
      x_train_clients.append(x_trainset)
      y_train_clients.append(y_trainset)

      #print("x_train_clients", x_trainset.shape)
      #print("y_train_clients", y_trainset.shape)


    return x_train_clients, y_train_clients, x_test, y_test


def write_results(file_dir, string):
    with open(f"{file_dir}/unlearning_result.txt", "a+") as result_file:
        result_file.write(string)


def evaluate_avg_std(values, unlearning_type, dataset, lr, epoch, num_run, client_id):

    run = len(values)
   
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


# Funzione per appiattire il dizionario
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_avg_std_results(result_file_dir, dict):
    
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

    csv_path = f"{result_file_dir}/unlearning_results.csv"

    data = flatten_dict(dict)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(data.keys())
        
        writer.writerow(data.values())

    """
    if not os.path.exists(csv_path):
        headers = ["unlearning_type", "dataset", "lr", "epoch", "run", "client"
                   "avg_acc_forgetting_original", "avg_acc_forgetting_retrained", "avg_acc_forgetting_unlearning", "avg_acc_forgetting_unlearning_after_recovery",
                   "std_acc_forgetting_original", "std_acc_forgetting_retrained", "std_acc_forgetting_unlearning", "std_acc_forgetting_unlearning_after_recovery",
                   "avg_acc_test_original", "avg_acc_test_retrained", "avg_acc_test_unlearning", "avg_acc_test_unlearning_after_recovery",
                   "std_acc_test_original", "std_acc_test_retrained", "std_acc_test_unlearning", "std_acc_test_unlearning_after_recovery",
                   "avg_mia_original", "avg_mia_retrained", "avg_mia_unlearning", "avg_mia_unlearning_after_recovery",
                   "std_mia_original", "std_mia_retrained", "std_mia_unlearning", "std_mia_unlearning_after_recovery",
                   "avg_recovery_round", "stf_recovery_round"]
        pd.DataFrame.to_csv(path_or_buf=csv_path, mode='w', header=[])

        return
    pd.DataFrame.to_csv(path_or_buf=csv_path, mode='a', header=False)
    """

    return

def main(num_clients, unlearning_client, only_unlearning_client, num_local_classes, 
         dataset_name, result_file_dir, learning_rate, epoch,
         natural_baseline=False, mia_avg=0,):

    #write_results(result_file_dir, f"\n\nParameters: Num_clients {num_clients}, num_local_classes {num_local_classes}, only_unlearning_client {only_unlearning_client}, \n"\
    #                  f"unlearning_client {unlearning_client}, unlearning_distilled_method {unlearning_distilled_method}, client_n_dd {client_n_dd}, \n"\
    #                  f"learning_rate {learning_rate}, epoch {epoch}, factor {factor}, natural_baseline {natural_baseline}, mia_avg {mia_avg}, fed_quit {fed_quit}\n")
    
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

    
    print("Not distillation")
    x_train, y_train, x_test, y_test = divide_data(num_client=num_clients, num_local_class=num_local_classes, dataset_name=dataset_name, i_seed=0, num_classes=10)
    z_train_client_unlearning = np.array([])
    if only_unlearning_client:
        # load unlearning client's (client UNLEARNING_CLIENT) data
        x_train_client_unlearning = x_train[unlearning_client]
        y_train_client_unlearning = y_train[unlearning_client]
        z_train_client_unlearning = np.full(y_train_client_unlearning.shape, 1)
    else:
        x_train_client_unlearning = x_train
        y_train_client_unlearning = y_train
        x_train_client_unlearning = np.array(x_train)
        y_train_client_unlearning = np.array(y_train)
        #For i < unlearning_client -> z = 0
        for i in range(0, unlearning_client):
            print(i)
            z_train_client_unlearning = np.append(z_train_client_unlearning, np.full(y_train[i].shape, 0))
        # unlearning_client -> z = 1
        z_train_client_unlearning = np.append(z_train_client_unlearning, np.full(y_train[unlearning_client].shape, 1))
        # for i > unlearning_client -> z = 0
        for i in range(unlearning_client+1, num_clients):
            print(i)
            z_train_client_unlearning = np.append(z_train_client_unlearning, np.full(y_train[i].shape, 0))
    
        # shape (10, 6000, 28, 28, 1) -> (6000, 28, 28, 1)
        x_train_client_unlearning = x_train_client_unlearning.reshape(-1, 28, 28, 1)
        # shape (10, 6000) -> (6000,)
        y_train_client_unlearning = y_train_client_unlearning.reshape(-1)
    ds_train_client_unlearning = tf.data.Dataset.from_tensor_slices((x_train_client_unlearning, y_train_client_unlearning, z_train_client_unlearning))

    # pre-processing
    ds_train_client_unlearning = ds_train_client_unlearning.map(lambda x, y, z: (x, y, tf.cast(z, tf.int32)))
    ds_train_client_unlearning = ds_train_client_unlearning.shuffle(1032).batch(32)
    
    # crei le combinazioni di hyperparametri
    """original_model = tf.keras.models.load_model('original_model.keras')
    original_model_clone = create_cnn_model()
    original_model_clone.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    original_model_clone.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    original_model_clone.set_weights(original_model.get_weights())"""
    #unlearning_model = UnlearningModel(original_model_clone) #Versione che usa solo la KL divergence sui forgetting data
    unlearning_model = UnlearningModelRandomLabel(original_model_clone)
    #unlearning_model.compile(optimizer="adam", metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    # sarebbe la nostra baseline di unlearning
    #unlearning_model.fit(ds_train_client_unlearning, epochs=1)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True,)
    
    unlearning_model.compile(optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

    if not natural_baseline:
        unlearning_model.fit(ds_train_client_unlearning, epochs=epoch, callbacks=[callback])
    
    
    x_train_client_unlearning = x_train[unlearning_client]
    y_train_client_unlearning = y_train[unlearning_client]
    forget_ds = tf.data.Dataset.from_tensor_slices((x_train_client_unlearning, y_train_client_unlearning,))

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # print((np.vstack(x_train[1:])).shape)
    # print((np.concatenate(y_train[1:])).shape)

    indexes = list(range(num_clients))
    indexes.pop(unlearning_client)

    ds_retain = tf.data.Dataset.from_tensor_slices(
        (np.vstack([x_train[i] for i in indexes]),
        np.concatenate([y_train[i] for i in indexes]))
    )
    # 54000 is the maximum len of retain dataset
    # la dimensione per il buffer di shuffle deve essere almeno la lunghezza max del tensore
    ds_retain = ds_retain.shuffle(54000, reshuffle_each_iteration=False)


    forget_ds = forget_ds.shuffle(1032).batch(32)
    # note reshuffle each False
    # la dimensione per il buffer di shuffle deve essere almeno la lunghezza max del tensore
    ds_test = ds_test.shuffle(1032*10, reshuffle_each_iteration=False)

    # evaluation - original model
    #original_model = tf.keras.models.load_model('original_model.keras')
    results = original_model.evaluate(forget_ds, verbose=2)
    accuracy_on_forgetting_original = results[1]
    results = original_model.evaluate(ds_test.batch(32), verbose=2)
    accuracy_on_test_original = results[1]
    #mia = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=forget_ds, model=original_model)
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
    #mia = evaluate_mia(ds_retain.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_test.take(NUM_EXAMPLES_MIA_TUNING).batch(32), ds_forgetting=forget_ds, model=retrained_model)
    print("Retrained Model")
    print(f"accuracy_on_forgetting: {accuracy_on_forgetting_retrained} -- accuracy_on_test: {accuracy_on_test_retrained} -- mia: {mia_retrained}")
    write_results(".",f"Retrained Model\t accuracy_on_forgetting: {accuracy_on_forgetting_retrained} -- accuracy_on_test: {accuracy_on_test_retrained} -- mia: {mia_retrained}\n")

    # evaluation - unlearning model
    #unlearning_model = tf.keras.models.load_model('new_model.keras')
    #print(1)
    unlearning_model_to_evaluate = create_cnn_model()
    unlearning_model_to_evaluate.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    unlearning_model_to_evaluate.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    unlearning_model_to_evaluate.set_weights(unlearning_model.get_weights())
    results = unlearning_model_to_evaluate.evaluate(forget_ds, verbose=2)
    #print(results)
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
        NUM_EXAMPLES_MIA_TUNING,
        natural_baseline,
        mia_avg
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

    

        
    num_clients = args.sys_n_client
    unlearning_client = args.sys_n_unlearning_client
    only_unlearning_client = args.sys_unlearning_only_one_client
    num_local_classes = args.sys_n_local_class #10: IID, 2: NON-IID
    dataset_name = args.sys_dataset
    model_name = args.sys_model
    result_file_dir = args.sys_dir

    learning_rates = [0.001, 0.0001, 0.00001]
    epochs = [1, 5, 30]
    
    unlearning_distilled_method = True
    only_unlearning_client = False
    natural_baseline = False
    mia_avgs = []
    
    learning_rate = 0
    epoch = 0
    
    for unlearning_client in range(3, 5): #5
        for i in range(3):
            main(num_clients,
                unlearning_client,
                only_unlearning_client, # only_unlearning_client=True, -> solo il client in unlearning
                num_local_classes,
                dataset_name,
                result_file_dir,
                learning_rate,
                epoch,
                natural_baseline=natural_baseline,
                mia_avg=mia_avgs[unlearning_client])

# python3 unlearning.py -nc 10 -nuc 1 -uoc True -ck 5 -ds fashion -md cnn -dir .
# python3 unlearning.py -nc 10 -nuc 1 -uoc false -ck 5 -ds fashion -md cnn -dir . start 20:25
# python3 unlearning.py -nc 10 -nuc 3 -uoc True -ck 5 -ds fashion -md cnn -dir .