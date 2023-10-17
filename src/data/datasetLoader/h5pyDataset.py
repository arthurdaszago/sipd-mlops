from os.path import exists
import stat
import h5py
import numpy as np
import os
import logging

class H5pyDataset:
    def __init__(self, data=None, labels=None, classes_dict=None, file_name=None):
      self.X = data
      self.y = labels
      self.classes_dict = classes_dict
      self.file_name = file_name
    
    def __verify_file_name(self):
      if exists(self.file_name):
        _id = 1
        while (exists('%s_%s' % (self.file_name, str(_id)))):
          _id+=1

        self.file_name = ("%s_%s" % (self.file_name, str(_id)))      

    def saveH5pyFile(self):
      if self.X is None:
        logging.error(("'data' parameter must be informed. Process aborted!"), exc_info=True)
        raise ValueError

      if self.y is None:
        logging.error(("'labels' parameter must be informed. Process aborted!"), exc_info=True)
        raise ValueError

      if self.classes_dict is None:
        logging.error(("'classes_dict' parameter must be informed. Process aborted!"), exc_info=True)
        raise ValueError

      if self.file_name is None:
        logging.error(("'file_name' parameter must be informed. Process aborted!"), exc_info=True)
        raise ValueError
      
      self.__verify_file_name()

      logging.info(("Saving dataset as %s. . ." % self.file_name), exc_info=True)
      logging.info(self.classes_dict, exc_info=True)


      try:
        hf = h5py.File(self.file_name, 'w')
      except OSError as e:
        logging.error("%s" % e, exc_info=True)
        raise e
      else:
        for sub in self.X:
          hf.create_group(sub)
          hf[sub].create_dataset('X', data=self.X[sub])
          hf[sub].create_dataset('y', data=self.y[sub])

        _dict_group = hf.create_group('classes_dict')
        for k, v in self.classes_dict.items():
          _dict_group[str(k)] = v.encode()

        hf.close()

      try:
        os.chmod((self.file_name), mode=0o664)
      except Exception as e:
        logging.error("%s" % e, exc_info=True)
        raise e
      
      return self.file_name