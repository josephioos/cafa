from lib.datasets import cifar10, crossset
import numpy as np

shared_config = {
    "iteration" : 100000,
    "warmup" : 40000,
    "lr_decay_iter" : 80000,
    "lr_decay_factor" : 0.2,
    "batch_size" : 100,
}
### dataset ###
cifar10_config = {
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 6,
}

crossset_config = {
    "transform" : [True, True, True],
    "dataset" : crossset.CROSSSET,
    "num_classes" : 20,
}

### algorithm ###
vat_config = {
    # virtual adversarial training
    "xi" : 1e-6,
    "eps" : {"cifar10":6, "svhn":1, "crossset":6, "cifar100":6, "imagenet32":6, "tinyimagenet":6},
    "consis_coef" : 0.3,
    "lr" : 3e-3
}
pl_config = {
    # pseudo label
    "threshold" : 0.95,
    "lr" : 3e-4,
    "consis_coef" : 1,
}
mt_config = {
    # mean teacher
    "ema_factor" : 0.95,
    "lr" : 4e-4,
    "consis_coef" : 8,
}
pi_config = {
    # Pi Model
    "lr" : 3e-4,
    "consis_coef" : 20.0,
}
mm_config = {
    # mixmatch
    "lr" : 3e-3,
    "consis_coef" : 8,
    "alpha" : 0.75,
    "T" : 0.5,
    "K" : 2,
}
fm_config = {
    # fixmatch
    "threshold" : 0.95,
    "lr" : 3e-4,
    "consis_coef" : 1,
}
supervised_config = {
    "lr" : 3e-3
}
### master ###
config = {
    "shared" : shared_config,
    "cifar10" : cifar10_config,
    "crossset" : crossset_config,
    "VAT" : vat_config,
    "PL" : pl_config,
    "MT" : mt_config,
    "PI" : pi_config,
    "MM" : mm_config,
    "FM" : fm_config,
    "supervised" : supervised_config
}
