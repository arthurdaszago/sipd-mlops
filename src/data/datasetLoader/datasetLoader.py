from src.model.cnnModelBuilder.jsonFileReading import JSONFileReading
from src.model.datasetLoader.h5pyDataset import H5pyDataset
from src.model.datasetLoader.imageDatasetLoader import ImageDatasetLoader
import os
import stat
from os.path import join, exists
from pathlib import Path
import logging

class DatasetLoader():
  def __init__(self, dataset_config_file_name, dataset_path, dataset_saving_path, dataset_h5py_filename):
    self.dataset_config_file_name = dataset_config_file_name
    self.dataset_path = dataset_path
    self.dataset_saving_path = dataset_saving_path
    self.dataset_h5py_filename = dataset_h5py_filename
    self.dataset_test_divided = None


  def __loadJSONConfigFile(self):
    try:
      _json_reader = JSONFileReading()
      _json_content = _json_reader.importJSONfile(self.dataset_config_file_name)
    except OSError as e:
      logging.error(("%s" % e), exc_info=True)
      raise e
    else:
      self.dsize = tuple(_json_content['input_shape'])
      self.dsize = (self.dsize[0], self.dsize[0])
      self.classes_to_load = _json_content['classes']
      self.dataset_test_divided = True if _json_content['dataset_test_divided'] == "True" else False

  def __save(self):
    if not exists(self.dataset_saving_path):
      try:
        os.makedirs(self.dataset_saving_path, mode=stat.S_IROTH, exist_ok=True)
      except OSError as e:
        logging.error(("%s" % e), exc_info=True)
        raise e

 
    _h5py_file = join(self.dataset_saving_path, self.dataset_h5py_filename)
    _h5py_dataset = H5pyDataset(self.dataset.loaded_images, self.dataset.loaded_classes, self.dataset.classes_dict, _h5py_file)
    return _h5py_dataset.saveH5pyFile()


  def main(self):
    self.__loadJSONConfigFile()
    self.dataset = ImageDatasetLoader(self.dataset_path, self.classes_to_load, self.dsize, dataset_test_divided=self.dataset_test_divided)
    try:
      self.dataset.loadImagesFromDatasetPath()
    except Exception as e:
      logging.error(("%s" % e), exc_info=True)
      raise e
    else:
      return self.__save()