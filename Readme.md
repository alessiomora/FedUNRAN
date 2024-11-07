# FedUNRAN

## Requirements
To run this project, you'll need:

* Python
* Tensorflow 

## Installation
To install the required libraries, you can use pip:

```shell
pip install tensorflow
```

## Usage
Once you have Python and the required libraries installed, you can run the code by following these steps.

### Step 1 - Running fedAVG.py
You can run the code using the following command:

```shell
python3 fedAVG.py -nc 10 -nuc 3 -ck 5 -ds fashion -md cnn -nr 5 -ne 1 -dir .
```

#### Explanations of Arguments
- `--sys-n_client` `-nc`: Number of clients
- `--sys-n_unlearning_client` `-nuc`: Index of unlearning client
- `--sys-n_local_class` `-ck`: Number of classes in each client
- `--sys-dataset` `-ds`: Dataset name (one of "mnist", "cifar10", "fashion")
- `--sys-model` `-md`: Model name (one of "cnn", "resnet50", "resnet50_pretrained", "alexnet")
- `--sys-n_round` `-nr`: Number of global communication rounds
- `--sys-n_epochs` `-ne`: Number of server epochs
- `--sys-dir` `-dir`: Directory in which to save the results. If it does not exist, a file named fedAVG_result.txt will be created, otherwise, the new results will be appended.

The code generates and saves two models in the path provided to -dir: original_model_{Index of unlearning client}.keras and retrained_model_{Index of unlearning client}.keras.
The first is created by applying the fedAvg algorithm considering all clients (parameter -nc) and running for a number of rounds equal to -nr.
The second, on the other hand, is created by applying the fedAvg algorithm considering all clients (parameter -nc) excluding the client with index equals to the parameter -nuc and running for a number of rounds equal to -nr.

Finally, a file fedAVG_result.txt is created where the accuracy results of the two models are saved.

### Step 2 - Running unlearning.py
You can run the code using the following command:

```shell
python3 unlearning.py -nc 10 -nuc 3 -ck 5 -ds fashion -md cnn -nr 3 -ne 30 -lr 0.001 -dir .
```

#### Explanations of Arguments
- `--sys-n_client` `-nc`: Number of clients
- `--sys-n_unlearning_client` `-nuc`: Index of unlearning client
- `--sys-n_local_class` `-ck`: Number of classes in each client
- `--sys-dataset` `-ds`: Dataset name (one of "mnist", "cifar10", "fashion")
- `--sys-model` `-md`: Model name (only "cnn")
- `--sys-n_round` `-nr`: Number of global communication rounds
- `--sys-n_epochs` `-ne`: Number of unlearning epochs
- `--sys-lr` `-lr`: Learning rate used in the unlearning method
- `--sys-dir` `-dir`: Directory in which to save the results. If it does not exist, a file named fedUNRAN_results.txt and a file fedUNRAN_results.csv will be created, otherwise, the new results will be appended.


The code loads the orginal_model and the retrained_model created at step 1. The unlearning_model is created by applying the FedUNRAN algorithm to a copy of the original_model. Subsequently, all three models are evaluated and compared. Finally, the FedAVG algorithm is applied to the unlearning_model to observe after how many rounds it achieves performance similar to that of the retrained_model.

At the end, two files (fedUNRAN_results.txt and fedUNRAN_results.csv) are created where all the results of the three models are saved.
