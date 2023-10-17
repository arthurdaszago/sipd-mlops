import sys
from cnnModelBuilder.src.cnnModelBuilder import CNNModelBuilder
import sys
import logging
import numpy as np

class CNNModelPrediction:
  def __init__(self, cnn_config_file=None, batch_size=128):
    self.cnn_config_file = cnn_config_file
    self.batch_size = batch_size

    self.__loadCNNModel()


  def __loadCNNModel(self):
    self.model = CNNModelBuilder()

    try:
      self.model.loadJSONConfigFile(self.cnn_config_file)
    except FileNotFoundError as e:
      logging.error(("FileNotFoundError in CNNModelPrediction. CNN config file not found, please check both path and filename: %s" % e), exc_info=True)
      raise e

    try:
      self.model.loadModel()
    except FileNotFoundError as e:
      logging.error(("FileNotFoundError in CNNModelPrediction. CNN model file not found, please check both path and filename in configuration file: %s" % e), exc_info=True)
      raise e


  def runPrediction(self, x_batch=None):
    if x_batch is None:
        logging.warning("Warning: x_batch param cannot be 'None' for model prediction. Please check it!")
        return None
    
    return self.model.cnn_model.predict(np.array(x_batch).astype(np.float32), self.batch_size)