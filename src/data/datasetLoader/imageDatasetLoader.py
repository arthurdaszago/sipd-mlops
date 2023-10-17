import os, sys
import cv2
import numpy as np
from os.path import isfile, join
from src.model.imageLoader.imageLoader import ImageLoader
from src.model.dicomLoader.dicomLoader import DicomLoader
from pydicom.errors import InvalidDicomError
import logging

class ImageDatasetLoader:
  def __init__(self, dataset_path, classes_to_load, dsize=(244,244), interpolation = cv2.INTER_LINEAR, dataset_test_divided=False):
    self.loaded_images = {}
    self.loaded_classes = {}
    self.dsize = dsize
    self.listed_files = {}    
    self.classes_dict = {}
    self.dataset_path = dataset_path
    self.classes_to_load = classes_to_load
    self.interpolation = interpolation
    self.dataset_test_divided = dataset_test_divided


  def __listDirPath(self):
    if self.dataset_test_divided is False:
      _subDatasets = ["train"]
    else:
      try:
        _subDatasets = os.listdir(self.dataset_path)
      except Exception as e:
        logging.error("Error: %s" % e, exc_info=True)
        raise e
      
    for _sub in _subDatasets:
      self.listed_files.update({_sub:{}})
      try:
        if self.dataset_test_divided is False:
          _dirs = os.listdir(join(self.dataset_path))
        else:
          _dirs = os.listdir(join(self.dataset_path, _sub))
      except Exception as e:
        logging.error("Error: %s" % e, exc_info=True)
        raise e
      else:
        for _dir in _dirs:
          if self.dataset_test_divided is False:
            self.listed_files[_sub].update({_dir: 
                            [_f for _f in os.listdir(join(self.dataset_path, _dir)) if isfile(join(self.dataset_path, _dir, _f ))]
                          })
          else:
            self.listed_files[_sub].update({_dir: 
                            [_f for _f in os.listdir(join(self.dataset_path, _sub, _dir)) if isfile(join(self.dataset_path, _sub, _dir, _f ))]
                          })
                      
      
  def loadImagesFromDatasetPath(self):
    self.__listDirPath()
    
    try:
      for sub in self.listed_files:
        self.loaded_images.update({sub:[]})
        self.loaded_classes.update({sub:[]})

        for i, c in enumerate(self.classes_to_load):
          # if sub == 'train':
          self.classes_dict[i] = c
          logging.info("Loading '%s'(%i) class from '%s' subset . . ." % (c, i, sub), exc_info=True)
          print("Loading '%s'(%i) class from '%s' subset . . ." % (c, i, sub))

          for f in self.listed_files[sub][c]:
            try:
              if self.dataset_test_divided is False:
                _img = self.__loadImageFromRepository(join(self.dataset_path, c), f, self.dsize)
              else:
                _img = self.__loadImageFromRepository(join(self.dataset_path, sub, c), f, self.dsize)
                 
            except Exception as e:
                logging.warning("%s" % e, exc_info=True)
                pass
            else:
              if _img is not None:
                self.loaded_images[sub].append(_img)
                self.loaded_classes[sub].append(i)

    except Exception as e:
      raise e
    else:
      for sub in self.loaded_images:
        self.loaded_images[sub] = np.array(self.loaded_images[sub]).astype(np.float32)
        self.loaded_classes[sub] = np.array(self.loaded_classes[sub]).astype(int)

      return self.loaded_images, self.loaded_classes, self.classes_dict


  def __loadImageFromRepository(self, repository_files, image_filename, img_input_shape):
      import mimetypes
      from os.path import join
      _image_file = join(repository_files, image_filename)
      _mimeType = mimetypes.guess_type(_image_file)
      
      if _mimeType[0] == 'image/png' or _mimeType[0] == 'image/jpeg':
          try:
              _img = ImageLoader(dsize=img_input_shape).loadImage(image_file=_image_file)
          except FileNotFoundError as e:
              raise e
          except TypeError as e:
              raise e
          except Exception as e:
              raise e
          else:
              logging.info("Susscceful loaded image: %s. Type: %s" % (_image_file, str(_mimeType[0])), exc_info=True)
              return _img
      else:
          try:
              _img = DicomLoader(dsize=img_input_shape).loadDicomImage(image_file=_image_file)
          except FileNotFoundError as e:
              raise e
          except InvalidDicomError as e:
              raise e
          except Exception as e:
              raise e
          else:
              logging.info(("Susscceful loaded image: %s. Type: 'DCM'" % _image_file), exc_info=True)
              return _img
