import numpy as np
import json

class Config(object):
    def __init__(self, d):
        self.__dict__ = d

class TrainingCounter(object):
    def __init__(self, config, total_phrase_num):
        self.config = config
        self.count = 0
        self.best_loss = np.inf
        self.epoch = 0
        self.best_epoch = 0
        self.epoch_loss = 0
        self.epoch_size = int(np.ceil(total_phrase_num/self.config.batch_size))
        self.no_improvement = 0
        self.stop = False

    # call this method to increment the counter after each iteration
    def __call__(self, loss):
        self.count += 1
        self.epoch_loss += loss
        if self.count % self.epoch_size == 0:
            self.epoch_end = True
            epoch_end_loss = self.epoch_loss / self.epoch_size
            print(f"Epoch: {self.epoch}, Loss: {epoch_end_loss}")
            if epoch_end_loss < self.best_loss:
                self.best_loss = epoch_end_loss
                self.best_epoch = self.epoch
                self.no_improvement = 0
            else:
                self.no_improvement += 1
                if self.no_improvement >= self.config.early_stopping:
                    self.stop = True
            self.epoch += 1
            self.epoch_loss = 0
        else:
            self.epoch_end = False

class TrainingArgs(object):
    def __init__(self, path_to_json):
        with open(path_to_json, 'r') as f:
            d = json.load(f)

        self.__dict__ = d

