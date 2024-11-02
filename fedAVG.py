import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
import random as rd
import numpy as np
from models.models import CNNModel
import argparse

def write_results(file_dir, string):
    with open(f"{file_dir}/fedAVG_result.txt", "a+") as result_file:
        result_file.write(string)

def fed_args():
    """
    Arguments for running FedD3
    :return: Arguments for FedD3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-nc', '--sys-n_client', type=int, required=True, help='Number of the clients')
    parser.add_argument('-nuc', '--sys-n_unlearning_client', type=int, required=True, help='Index of unlearning client')
    parser.add_argument('-ck', '--sys-n_local_class', type=int, required=True, help='Number of the classes in each client')
    parser.add_argument('-ds', '--sys-dataset', type=str, required=True, help='Dataset name, one of the following four datasets: MNIST, CIFAR-10, fashion, SVHN')
    parser.add_argument('-md', '--sys-model', type=str, required=True, help='Model name')
    parser.add_argument('-nr', '--sys-n_rounds', type=int, required=True, help='Number of rounds')
    parser.add_argument('-ne', '--sys-n_epochs', type=int, required=True, help='Number of epochs')
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


def fedavg(server_model, server_x_test, server_y_test, clients_x_train, clients_y_train, num_rounds=1, epochs=1, num_clients=10, retrained=False, unlearning_client=1, result_file_dir="."):
    
    
    print("Server model - Initial evaluate")
    test_loss, test_accuracy,sparse_categorical_crossentropy = server_model.evaluate(server_x_test, server_y_test)
    n = 0
    random_client_indexs = list(range(num_clients))
    model_type = "original model"
    if retrained:
        random_client_indexs.pop(unlearning_client)
        model_type = "retrained model"

    x_train_client_unlearning = clients_x_train[unlearning_client]
    y_train_client_unlearning = clients_y_train[unlearning_client]
    forget_ds = tf.data.Dataset.from_tensor_slices((x_train_client_unlearning, y_train_client_unlearning,))

    print(f"Client list: {random_client_indexs}")

    for i in range(num_clients):
        n += len(clients_x_train[i])
    
    write_results(result_file_dir, f"\nRun fedAVG on {model_type}\n")
    
    for round in range(num_rounds):
        print(f"Start federated training - round: {round}")
        #new_global_weights = np.zeros_like(server_model.get_weights(), dtype=object)
        new_global_weights = [np.zeros_like(ww) for ww in server_model.get_weights()]

        print("Select ", len(random_client_indexs), " clients")
        # fedAvg algorithm
        for index in random_client_indexs:
            # temp_global_weights = np.zeros_like(server_model.get_weights(), dtype=object)
            client_x_train = clients_x_train[index]
            client_y_train = clients_y_train[index]

            client_weights_list, num_samples = client_update(server_model, index, client_x_train, client_y_train, round, epochs=epochs)
            temp_global_weights = [ww * (num_samples/n) for ww in client_weights_list]

            new_global_weights = [new_global_weights[i] + temp_global_weights[i]  for i in range(len(new_global_weights))]

        server_model.set_weights(new_global_weights)

        server_model.set_weights(new_global_weights)
        print(f"Evaluation - round: {round}")
        print("Test set ")
        test_loss, test_accuracy, sparse_categorical_crossentropy = server_model.evaluate(server_x_test, server_y_test)
        
        write_results(result_file_dir, f"Round {round} - {model_type}: Test set \t test_loss: {test_loss}  \t test_acc {test_accuracy} \t sparse_categorical_crossentropy {sparse_categorical_crossentropy}\n")
        print("Forget set ")
        test_loss, test_accuracy, sparse_categorical_crossentropy = server_model.evaluate(forget_ds.batch(32))
        write_results(result_file_dir, f"Round {round} - {model_type}: Forget set \t test_loss: {test_loss}  \t test_acc {test_accuracy} \t sparse_categorical_crossentropy {sparse_categorical_crossentropy}\n")
        
    write_results(result_file_dir, "\n")
    return server_model


def client_update(server_model, client_index, client_x_train, client_y_train, round=0, epochs=1):
    
    print(f"Update client {client_index} - round: {round}")
    client_model = create_cnn_model()
    client_model.predict(np.random.random((1, 28, 28, 1)))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # qui occhio a usare sparse_categorical_accuracy perchè non prederebbe from_logits=True
    client_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    client_model.set_weights(server_model.get_weights())

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

def main():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # If no GPUs are avaible, the CPU is automatically used
    with tf.device('/GPU:0'):

        # main
        #NUM_ROUNDS = 5
        #EPOCHS = 1
        #NUM_LOCAL_CLASSES = 5
        #UNLEARNING_CLIENT = 1

        args = fed_args()
        
        num_clients = args.sys_n_client
        unlearning_client = args.sys_n_unlearning_client
        num_local_classes = args.sys_n_local_class #10: IID, 2: NON-IID
        dataset_name = args.sys_dataset
        model_name = args.sys_model
        num_rounds = args.sys_n_rounds
        num_epochs = args.sys_n_epochs
        result_file_dir = args.sys_dir

        write_results(result_file_dir, f"\nResult for {dataset_name} using {model_name}:\n")
        write_results(result_file_dir, f"Num_clients {num_clients}, num_local_classes {num_local_classes}, num_round {num_rounds},"\
                      f" unlearning_client {unlearning_client}, num_epoch {num_epochs}\n")

        # prepare data
        x_train, y_train, x_test, y_test = divide_data(num_client=10, num_local_class=num_local_classes, dataset_name=dataset_name, i_seed=0, num_classes=10)

        print(f"Composition of client datasets")
        for i in range(len(y_train)):
            unique, counts = np.unique(y_train[i], return_counts=True)
            print(f"\t client {i}: {dict(zip(unique, counts))}")

        # original model training
        print("Original Model Training")
        original_model = create_cnn_model()
        original_model.predict(np.random.random((1, 28, 28, 1)))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # qui occhio a usare sparse_categorical_accuracy perchè non preNderebbe from_logits=True
        original_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
        # original_model.fit(server_x_train, server_y_train, epochs=num_epochs)
        original_model = fedavg(original_model,
                                x_test,
                                y_test,
                                x_train,
                                y_train,
                                num_rounds=num_rounds,
                                epochs=num_epochs,
                                retrained=False,
                                unlearning_client=unlearning_client,
                                result_file_dir=result_file_dir
                                )

        # retrained model training
        print("Retrained Model Training")
        retrained_model = create_cnn_model()
        retrained_model.predict(np.random.random((1, 28, 28, 1)))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # qui occhio a usare sparse_categorical_accuracy perchè non prederebbe from_logits=True
        retrained_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
        # original_model.fit(server_x_train, server_y_train, epochs=num_epochs)
        retrained_model = fedavg(retrained_model,
                                x_test,
                                y_test,
                                x_train,
                                y_train,
                                num_rounds=num_rounds,
                                epochs=num_epochs,
                                retrained=True,
                                unlearning_client=unlearning_client,
                                result_file_dir=result_file_dir
                                )
        name = f"original_model_{unlearning_client}.keras"
        original_model.save(name)
        name = f"retrained_model_{unlearning_client}.keras"
        retrained_model.save(name)
        return
        
   
if __name__ == "__main__":
    main()

"""
USAGE:
- python3 fedAVG.py -nc 10 -nuc 1 -ck 5 -ds fashion -md cnn -nr 5 -ne 1 -dir .
 python3 fedAVG.py -nc 10 -nuc 3 -ck 5 -ds fashion -md cnn -nr 5 -ne 1 -dir .  
"""