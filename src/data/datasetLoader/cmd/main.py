from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
print(sys.path)

from model.datasetLoader.datasetLoader import DatasetLoader
import sys, getopt


def main(argv):
  path_root = str(Path(__file__).parents[3])
  dataset_loading_config_file = ("%s/src/config/dataset_sars_loading_config_file.json" % path_root)

  try:
    _, args = getopt.getopt(argv[1:], "f")
    try:
      dataset_loading_config_file = args[0]
    except IndexError:
      print("usage: %s -f [path to] dataset_loading_config_file.json" % argv[0])    
  except getopt.GetoptError as error:
      print("usage: %s -f [path to] dataset_loading_config_file.json" % argv[0])
  else:

    print("Using '%s' config file!" % dataset_loading_config_file)

    datasetLoader = DatasetLoader(dataset_loading_config_file)
    datasetLoader.run()

#    print(datasetLoader.dataset.loaded_images.shape)



if __name__ == "__main__":
    main(sys.argv)