"""MAST: masked self-supervised pretraining for time series forecasting.

Two stages, both usable from the command line:

* ``pretrain.py``  - self-supervised pretraining on ETTh1.
* ``forecast.py``  - downstream forecasting via linear probing or end-to-end finetuning.
"""

__all__ = ['MAST', 'MASTModel', 'Learner', 'get_dls']
