import cv2
from os.path import isfile, join
import logging

class ImageLoader:
  def __init__(self, dsize=(244,244), interpolation = cv2.INTER_LINEAR):
    self.dsize = dsize
    self.interpolation = interpolation


  def loadImage(self, image_file):
    if not isfile(image_file):
      logging.error(("FileNotFoundError: %s" % image_file), exc_info=True)
      raise FileNotFoundError(image_file)

    _img = cv2.imread(image_file)

    if _img is not None:
      try:
        _img = cv2.resize(_img, dsize=self.dsize, interpolation=self.interpolation)
      except Exception as e:
        logging.error("Exception: cv2 resizing", exc_info=True)
        raise e("cv2 resizing")
      else:
        return _img
    else:
      logging.error("Image TypeError: None", exc_info=True)
      raise TypeError
