from pathlib import Path
from os.path import join, exists
import os
import sys
import tensorflow as tf
import logging

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
CNN_PATH_ROOT = os.environ.get('CNN_PATH_ROOT')

if CNN_PATH_ROOT is None:
    CNN_PATH_ROOT = str(Path(__file__).parents[1])
    os.environ['CNN_PATH_ROOT'] = CNN_PATH_ROOT
    logging.warning(("Environment variable 'CNN_PATH_ROOT' not found. SIPD-CNN started using %s path root." % CNN_PATH_ROOT), exc_info=True)

sys.path.append(CNN_PATH_ROOT)
from src.data.cnnModelBuilder.cnnModelBuilder import CNNModelBuilder
from src.data.datasetLoader.datasetLoader import DatasetLoader

class Train:
    def __init__(self):
        self.LOGGING_FILE = os.environ.get('LOGGING_FILE')
        if self.LOGGING_FILE is None:
            logging.warning("Environment variable 'LOGGING_FILE' not found. Logging started without log file!", exc_info=True)
        else:
            _root_logger= logging.getLogger()
            _root_logger.setLevel(logging.INFO)
            _handler = logging.FileHandler(self.LOGGING_FILE, 'w', 'utf-8')
            _handler.setFormatter(logging.Formatter('%(asctime)s %(name)s:%(levelname)s:%(message)s'))
            _root_logger.addHandler(_handler)

        self.CNN_CONFIG_FILE = os.environ.get('CNN_CONFIG_FILE')
        self.CNN_PATH_ROOT = os.environ.get('CNN_PATH_ROOT')
        self.TRAINING_DATASET = os.environ.get('TRAINING_DATASET')
        self.H5_DATASET_PATH = os.environ.get('H5_DATASET_PATH')
        self.H5_DATASET_FILENAME = os.environ.get('H5_DATASET_FILENAME')
        self.CNN_MODEL_PATH = os.environ.get('CNN_MODEL_PATH')
        self.CNN_STATS_PATH = os.environ.get('CNN_STATS_PATH')    

    def createModelPaths(self):         

      if not exists(self.H5_DATASET_PATH):
        try:
            os.makedirs(self.H5_DATASET_PATH, mode=0o777, exist_ok=True)
        except Exception as e:
            logging.error("Error: %s" % e, exc_info=True)
            raise e

      if not exists(self.CNN_MODEL_PATH):
        try:
            os.makedirs(self.CNN_MODEL_PATH, mode=0o777, exist_ok=True)
        except Exception as e:
            logging.error("Error: %s" % e, exc_info=True)
            raise e

      if not exists(self.CNN_STATS_PATH):
        try:
            os.makedirs(self.CNN_STATS_PATH, mode=0o777, exist_ok=True)
        except Exception as e:
            logging.error("Error: %s" % e, exc_info=True)
            raise e


    def main(self):
      try:
        self.createModelPaths()
      except Exception as e:
        logging.error("Error when trying to create model paths: %s" % e, exc_info=True)
        raise e

      dsAsH5pyFileName = join(self.H5_DATASET_PATH, self.H5_DATASET_FILENAME) 

      if not exists(join(dsAsH5pyFileName)):
        try:
          DatasetLoader(self.CNN_CONFIG_FILE,  self.TRAINING_DATASET, self.H5_DATASET_PATH, self.H5_DATASET_FILENAME).main()
        except Exception as e:
          logging.error("Error when trying to load dataset: %s" % e, exc_info=True)
          raise e
         

      cnn = CNNModelBuilder()
      cnn_config_file = self.CNN_CONFIG_FILE
      try:
        cnn.loadJSONConfigFile(cnn_config_file)
      except TypeError as e:
        raise e
      except FileNotFoundError as e:
        logging.error("File %s not found." % cnn_config_file, exc_info=True)
        raise e
      else:
        logging.info("JSON config file %s successful loaded" % cnn_config_file, exc_info=True)

      try:
        cnn.loadDatasetAsH5pyFile(dsAsH5pyFileName)
      except Exception as e:
        logging.error("Error when trying to load H5 dataset: %s" % dsAsH5pyFileName, exc_info=True)
        raise e
      else:
        logging.info("H5 dataset loaded from as: %s" % dsAsH5pyFileName, exc_info=True)
                  
      try:
        cnn.buildCNNModel()
        # logging.info("Summary: %s" % str(cnn.cnn_model.summary()), exc_info=True)
        cnn.runTraining()
      except Exception as e:
        logging.error("Error when trying to build cnn model: %s" % e, exc_info=True)
        raise e
      else:
        logging.info("CNN model was successful trained", exc_info=True)
         

      training_graph_file = join(self.CNN_STATS_PATH, ("%s_" % cnn.cnn_name))
      try:
        cnn.plotAccuracyGraph(cnn.training_history, training_graph_file)
        cnn.plotLossGraph(cnn.training_history, training_graph_file)
      except Exception as e:
        logging.error("Error when trying to plot model history: %s" % e, exc_info=True)
        raise e
         
      try:
        _prediction = cnn.runTestPrediction()
      except Exception as e:
        logging.error("Error when trying to run the test prediction: %s" % e, exc_info=True)
        raise e
         
      try:
        cnn.metricsEvaluation(_prediction, self.CNN_STATS_PATH)
      except Exception as e:
        logging.error("Error when trying to evaluate metrics: %s" % e, exc_info=True)
        raise e

      try:
        cnn.saveModel(self.CNN_MODEL_PATH)
      except Exception as e:
        logging.error("Error when trying to save the cnn model: %s" % e, exc_info=True)
        raise e

if __name__ == "__main__":
    train = Train()
    train.main()
