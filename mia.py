import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
import tensorflow as tf
#import tensorflow_datasets as tfds
#from omegaconf import DictConfig, OmegaConf
#import hydra
import os

"""
# from imagenet import get_x_y_from_data_dict
from basics_unlearning.dataset import get_string_distribution, \
    load_client_datasets_from_files, normalize_img, expand_dims
from basics_unlearning.model import create_cnn_model
"""


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(dataset, model):
    print("Collecting performance..")

    if dataset is None:
        return np.zeros([0, 10]), np.zeros([0])

    if dataset is not None:
        probs = []
        # labels = []

        prob = tf.nn.softmax(
            model.predict(dataset),
            axis=1
        )
        

        probs.append(prob)
        labels = np.concatenate([y for _, y in dataset], axis=0)
        # print(labels)
        return tf.concat(prob, axis=0).numpy(), labels
    else:
        return None, None


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    # MIA-efficacy: the number of forgetting data examples classified as non-training.

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)
        # target_train --> original train, retrained/resumed non-train
        # train 1
        # non train 0
        # low accuracy means most of the data are predicted to be non-training
        # original: should be high
        # retrained/resumed: should be low

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

        # target_train --> original train, retrained/resumed non-train
        # train 1
        # non train 0
        # low accuracy means most of the data are predicted to be training
        # original: should be low
        # retrained/resumed: should be high

    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    shadow_train_prob = torch.from_numpy(shadow_train_prob)
    shadow_train_labels = torch.from_numpy(shadow_train_labels)
    shadow_test_prob = torch.from_numpy(shadow_test_prob)
    shadow_test_labels = torch.from_numpy(shadow_test_labels)
    
    target_train_prob = torch.from_numpy(target_train_prob)
    target_train_labels = torch.from_numpy(target_train_labels)

    target_test_prob = torch.from_numpy(target_test_prob)
    target_test_labels = torch.from_numpy(target_test_labels)

    # NEW: necessario per far funzionare torch.gather
    shadow_train_labels = shadow_train_labels.to(torch.int64)
    shadow_test_labels = shadow_test_labels.to(torch.int64)
    target_train_labels = target_train_labels.to(torch.int64)
    target_test_labels = target_test_labels.to(torch.int64)

    
    shadow_train_corr = (
        torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    ).int()
    target_train_corr = (
            torch.argmax(target_train_prob, axis=1) == target_train_labels
        ).int()
    target_test_corr = (
        torch.argmax(target_test_prob, axis=1) == target_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    shadow_train_entr = entropy(shadow_train_prob)
    shadow_test_entr = entropy(shadow_test_prob)

    target_train_entr = entropy(target_train_prob)
    target_test_entr = entropy(target_test_prob)

    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    if target_train is not None:
        target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    else:
        target_train_m_entr = target_train_entr
    if target_test is not None:
        target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    else:
        target_test_m_entr = target_test_entr

    # acc_corr = SVC_fit_predict(
    #     shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    # )
    acc_corr = 0.0
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    # acc_entr = SVC_fit_predict(
    #     shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    # )
    acc_entr = 0.0
    # acc_m_entr = SVC_fit_predict(
    #     shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    # )
    acc_m_entr = 0.0
    # acc_prob = SVC_fit_predict(
    #     shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    # )
    acc_prob = 0.0
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
        "prob": acc_prob,
    }
    print(m)
    return m

def evaluate_mia(ds_retain, ds_test, ds_forgetting, model):
    print("Evaluating MIA")
    results_mia = SVC_MIA(shadow_train=ds_retain,
                        shadow_test=ds_test,
                        target_train=ds_forgetting,
                        target_test=None,
                        model=model)
    print(results_mia)
    return results_mia["confidence"]