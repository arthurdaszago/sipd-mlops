import math
import numpy as np
import tensorflow as tf

class BatchSequence(tf.keras.utils.Sequence):

    # def __init__(self, x, y, batch_size):
        # self.x = x
        # self.y = y
        # self.batch_size = batch_size
    def __init__(self, x_data, y_data, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        
    # def __len__(self):
    #     return math.ceil(len(self.x) / self.batch_size)
    def __len__(self):
        return int(np.ceil(len(self.x_data) / float(self.batch_size)))
    
    # def __getitem__(self, index):
    #     print("Index: %i" % index);
    #     low = index * self.batch_size
    #     # Cap upper bound at array length; the last batch may be smaller
    #     # if the total number of items is not a multiple of batch size.
    #     high = min(low + self.batch_size, len(self.x))
    #     batch_x = self.x[low:high]
    #     batch_y = self.y[low:high]

    #     return batch_x, batch_y
    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y