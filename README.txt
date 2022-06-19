# Univsersal Semi-Supervised Learning
This repository is reimplementation of [realistic-ssl-evaluation-pytorch]
Here is the original repo: https://github.com/perrying/realistic-ssl-evaluation-pytorch

# Requirements
- Python 3.6+
- PyTorch 1.1.0
- torchvision 0.3.0
- numpy 1.16.2

# How to run
1)download the visda2017 dataset to the corresponding path: ```./data/visda/```

2)run the following command ```python train_cafa.py -a [backbone method] -d [dataset] -tp [dataset settings]```

Default backbone method setting is PI. Please check the options by ```python train.py -h```

# Performance
visda dataset under subset mismatch using PI with one random run: 0.8805522322654724
visda dataset under intersectional mismatch using PI with one random run: 0.8585433959960938
