"""MAST: masked self-supervised pretraining for time series forecasting and classification.

Three entry points, all usable from the command line:

* ``pretrain.py``  - self-supervised pretraining on any supported dataset.
* ``forecast.py``  - downstream forecasting via linear probing or end-to-end finetuning.
* ``classify.py``  - downstream classification on the UEA datasets, same two modes.
"""

__all__ = ['MAST', 'MASTModel', 'Learner', 'get_dls']
