import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.errors import InvalidDicomError
from os.path import isfile
import numpy as np
import cv2
import logging

class DicomLoader():
    def __init__(self, dsize=(244,244), interpolation = cv2.INTER_LINEAR) -> None:
        self.dsize = dsize
        self.interpolation = interpolation

    def loadDicomImage(self, image_file):
        if not isfile(image_file):
            logging.error(("FileNotFoundError: %s" % image_file), exc_info=True)
            raise FileNotFoundError
        else:
            try:
                _ds = dicom.dcmread(image_file)
                _img_array = self.__readXRay(_ds)
                _img_array = self.__resizeImage(_img_array)
            except InvalidDicomError as e:
                logging.error(("Invalid dicom file '%s'" % image_file), exc_info=True)
                raise e
            except Exception as e:
                logging.error(("Cannot load '%s' dicom object" % image_file), exc_info=True)
                raise e("dicom error")
            else:
                return _img_array
    
    
    def __readXRay(self, ds):
        #https://www.kaggle.com/code/raddar/convert-dicom-to-np-array-the-correct-way
        try:
            data = apply_voi_lut(ds.pixel_array, ds)
        except Exception as e:
            logging.error(("Error when trying to aplly voi lut: %s" % e), exc_info=True)
            raise e

        if ds.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        return data

    def __resizeImage(self, img_array):
        if img_array is not None:
            try:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_array = cv2.resize(img_array, dsize=self.dsize, interpolation=self.interpolation)
            except Exception as e:
                logging.error(("Error in DicomLoader when trying to resize img_array with cv2: %s" % e), exc_info=True)
                raise e
            else:
                return img_array
        else:
            logging.error(("None Type Error in DicomLoader: %s" % e), exc_info=True)
            raise TypeError
