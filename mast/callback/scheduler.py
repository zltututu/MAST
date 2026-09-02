
__all__ = ['OneCycleLR']

from .core import Callback
from torch.optim import lr_scheduler



class OneCycleLR(Callback):
    def __init__(self, lr_max=None,
                        total_steps=None,                        
                        steps_per_epoch=None,
                        pct_start=0.3,
                        anneal_strategy='cos',
                        cycle_momentum=True,
                        base_momentum=0.85,
                        max_momentum=0.95,
                        div_factor=25.,
                        final_div_factor=1e4,
                        three_phase=False,
                        last_epoch=-1,
                        verbose=False):

        super().__init__()        
        self.lr_max = lr_max if lr_max else self.lr  
        self.total_steps, self.steps_per_epoch = total_steps, steps_per_epoch         
        self.pct_start = pct_start
        self.anneal_strategy, self.cycle_momentum = anneal_strategy, cycle_momentum
        self.base_momentum, self.max_momentum = base_momentum, max_momentum
        self.div_factor, self.final_div_factor = div_factor, final_div_factor
        self.three_phase = three_phase
        self.last_epoch = last_epoch
        self.verbose = verbose
                

    def before_fit(self):
        if not self.steps_per_epoch: self.steps_per_epoch = len(self.dls.train)
        self.lrs = []  # store lr values
        
        # Calculate total_steps if not provided
        if not self.total_steps:
            self.total_steps = self.n_epochs * self.steps_per_epoch
        
        self.scheduler = lr_scheduler.OneCycleLR(optimizer = self.opt, 
                                            max_lr = self.lr_max,
                                            total_steps = self.total_steps,
                                            pct_start=self.pct_start,
                                            anneal_strategy=self.anneal_strategy,
                                            cycle_momentum=self.cycle_momentum,
                                            base_momentum=self.base_momentum,
                                            max_momentum=self.max_momentum,
                                            div_factor=self.div_factor,
                                            final_div_factor=self.final_div_factor,
                                            three_phase=self.three_phase,
                                            last_epoch=self.last_epoch
                                            )

    def after_batch_train(self):
        if self.model.training: 
            self.scheduler.step()
            self.lrs.append( self.scheduler.get_last_lr()[0] )                  

    def after_fit(self):        
        self.learner.scheduled_lrs = self.lrs
