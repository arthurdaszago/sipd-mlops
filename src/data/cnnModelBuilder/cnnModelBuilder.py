import numpy as np
from src.model.cnnModelBuilder.jsonFileReading import JSONFileReading
from src.model.batchGenerator.batchGenerator import BatchSequence
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils
# import tensorflow.data.Dataset as Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from sklearn.model_selection import train_test_split
from os.path import join, exists
import os
import logging

class CNNModelBuilder:
    def __init__(self):
        # https://keras.io/guides/transfer_learning/

        self.config_file = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.cnn_name = None
        self.imput_shape = None
        self.classes = None
        self.pretrained_weights = None
        self.trainable_layers = None
        self.data_augmentation = None
        self.training_epochs = None
        self.batch_size = None
        self.optimizer = None
        self.learning_rate = None
        self.patience = None
        self.loss_function = None
        self.training_metrics = None
        self.classifier_layers = None
        self.cnn_model = None

        self.dataset_X = None
        self.dataset_y = None
        self.classes_dict = {}
        self.test_size = None

        self.training_history = None
        self.model_saving_path = None
        self.stats_saving_path = None

    def loadJSONConfigFile(self, config_file):
        self.config_file = config_file
        if self.config_file is None:
            logging.error("Error in loadJSONConfigFile:'config_file' parameter should be informed. Process aborted!", exc_info=True)
            raise FileNotFoundError(self.config_file)
        try:
            _json_file = JSONFileReading()
            _json_content = _json_file.importJSONfile(self.config_file)
        except FileNotFoundError as e:
            raise e
        else:
            self.cnn_name = _json_content['cnn_name']
            self.dataset_test_divided = True if _json_content['dataset_test_divided'] == "True" else False
#            self.stats_saving_path = _json_content['stats_saving_path']
#            self.model_saving_path = _json_content['model_saving_path']
            self.imput_shape = tuple(_json_content['input_shape'])
            self.classes = _json_content['classes']
            self.pretrained_weights = _json_content['pretrained_weights']
            self.trainable_layers = _json_content['trainable_layers']
            self.data_augmentation = True if _json_content['data_augmentation'] == "True" else False,
            self.training_epochs = _json_content['training_epochs']
            self.batch_size = None if _json_content['batch_size'] == "None" else int(_json_content['batch_size'])
            self.optimizer = _json_content['optimizer']
            self.learning_rate = None if _json_content['learning_rate'] == "None" else int(_json_content['learning_rate'])
            if _json_content['patience'] == "True":
                self.patience = True
            elif _json_content['patience'] == "False":
                self.patience = False
            else:
                self.patience = float(_json_content['patience'])
                
            self.loss_function = _json_content['loss_function']
            self.training_metrics = _json_content['training_metrics']
            self.classifier_layers = _json_content['classifier_layers']

    def loadCNNPreTrainedWeights(self):
        # https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
        pass

    def __datasetYInputPreprocess(self):       
        self.y_train = tf.keras.utils.to_categorical(self.y_train, len(self.classes_dict))
        self.y_test = tf.keras.utils.to_categorical(self.y_test, len(self.classes_dict))

        return self.y_train, self.y_test


    def buildImgAugmentationLayer(self):
        return Sequential(
            [
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomZoom(height_factor=0.5, width_factor=0.2),
                layers.RandomFlip(),
                layers.RandomCrop(height=self.imput_shape[0], width=self.imput_shape[1]),
                layers.RandomContrast(factor=0.1),
            ],
            name="img_augmentation"
        )

    def __unfreezeModel(self):
        try:
            if self.trainable_layers == "False":
                self.cnn_model._trainable = False
            elif self.trainable_layers == "True":
                self.cnn_model._trainable = True
            else:
                # Unfreeze the top X layers, according the self.trainable_layers, while leaving BatchNorm layers frozen
                for layer in self.cnn_model.layers[-int(self.trainable_layers):]:
                    if not isinstance(layer, layers.BatchNormalization):
                        layer.trainable = True
        except Exception as e:
            raise e("unfreeze")

    def __buildOptimizer(self):
        try:
            if self.optimizer is None:
                logging.error(("Error:'optimizer' parameter should be informed."), exc_info=True)            
                raise TypeError("optimizer is none")

            if self.optimizer == "Adam":
                if self.learning_rate is not None:
                    return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                else:
                    return tf.keras.optimizers.Adam()
            if self.optimizer == "Adagrad":
                if self.learning_rate is not None:
                    return tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
                else:
                    return tf.keras.optimizers.Adagrad()
            if self.optimizer == "Adadelta":
                if self.learning_rate is not None:
                    return tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate)
                else:
                    return tf.keras.optimizers.Adadelta()
        except Exception as e:
            raise e("optimizer")

    def __buildClassifierLayers(self):
        try:
            if self.classifier_layers is None:
                logging.error(("Error: 'classifier_layers' parameter should be not None."), exc_info=True)            
                raise TypeError("classifier_layers is none")

            if self.cnn_model is None:
                logging.error(("Error: 'cnn_model' does not instanced."), exc_info=True)            
                raise TypeError("cnn_model is none")

            _x = self.cnn_model.output

            for layer in self.classifier_layers:
                if layer['layer'] == 'GlobalAveragePooling2D':
                    _x = layers.GlobalAveragePooling2D(name=layer['name'])(_x)
                elif layer['layer'] == 'BatchNormalization':
                    _x = layers.BatchNormalization(name=layer['name'])(_x)
                elif layer['layer'] == 'Dropout':
                    _x = layers.Dropout(rate=layer['rate'], name=layer['name'])(_x)
                elif layer['layer'] == 'Dense' and layer['name'] == 'pred_layer':
                    _x = layers.Dense(units=len(self.classes_dict), activation=layer['activation'], name=layer['name'])(_x)
                else: # layer['layer'] == 'Dense':
                    _x = layers.Dense(units=layer['units'], activation=layer['activation'], name=layer['name'])(_x)
        except Exception as e:
            raise e("classifier layers")
        else:
            return _x

    def buildCNNModel(self):
        try:
            _input = layers.Input(shape=self.imput_shape)
            if self.data_augmentation is True:
                _input = self.buildImgAugmentationLayer()(_input)

            self.cnn_model = tf.keras.applications.efficientnet.EfficientNetB4(
                weights="imagenet",
                input_tensor = _input,
                include_top=False
            )
            
            self.__unfreezeModel()

            _output = self.__buildClassifierLayers()

            self.cnn_model = tf.keras.Model(_input, _output, name=self.cnn_name)

            _optimizer = self.__buildOptimizer()

            self.cnn_model.compile(
               optimizer=_optimizer, loss=self.loss_function, metrics=[self.training_metrics]
            )
        except Exception as e:
            raise e("build model")
        else:
            return self.cnn_model


    def runTraining(self):
        try:
            if self.patience is True:
                self.patience = self.training_epochs * 0.075
            elif (self.patience is False):
                pass
            else:
                self.patience = self.training_epochs * self.patience

            logging.error(("Classes_dict: %s" % str(self.classes_dict)), exc_info=True)
            logging.error(("Classes_dict len: %i" % len(self.classes_dict)), exc_info=True)

            if self.batch_size is not None:
                
                with tf.device("CPU"):                
                    _valTrain = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(self.x_train.shape[0], reshuffle_each_iteration=True).batch(self.batch_size)
                    _valTest = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).shuffle(self.x_test.shape[0], reshuffle_each_iteration=True).batch(self.batch_size)
#                    _valTest = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.batch_size)


            if self.patience is not False:
                _es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 2, patience = self.patience)
                self.training_history = self.cnn_model.fit(_valTrain,
                                                        validation_data = _valTest,
                                                        batch_size = None,
                                                        epochs = self.training_epochs,
                                                        callbacks = [_es])
            else:
                self.training_history = self.cnn_model.fit(_valTrain,
                                                        validation_data = _valTest,
                                                        batch_size = None,
                                                        epochs = self.training_epochs,
                                                        verbose=2)
        except Exception as e:
            raise e("training")
        else:
            return self.training_history


    def metricsEvaluation(self, y_pred, path_to_file):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score, precision_score, f1_score
        import json as jsn

        _file_name = self.cnn_name
        if not exists(path_to_file):
            try:
                os.makedirs(path_to_file, mode=0o740, exist_ok=True)
            except Exception as e:
                raise e
        
        _y_true = np.argmax(self.y_test, axis = 1)
        _cm = confusion_matrix(_y_true, y_pred)

        _stats = {
            "Accuracy": round(accuracy_score(_y_true, y_pred), 5),
            "Recall": round(recall_score(_y_true, y_pred, average='macro'), 5),
            "Precision": round(precision_score(_y_true, y_pred, average='macro'), 5),
            "F1-Score": round(f1_score(_y_true, y_pred, average='macro'), 5),
            "Classes": self.classes_dict
        }

        _t1 = ConfusionMatrixDisplay(_cm)
        _t1.plot()
        try:
            plt.savefig(join(path_to_file, ("%s_cm.jpg" %_file_name)))
        except Exception as e:
            raise e

        _stats = jsn.dumps(_stats, indent=4)
        try:
            with open(join(path_to_file, ("%s_stats.json" %_file_name)), "w") as outfile:
                outfile.write(_stats)
        except Exception as e:
            raise e

        _cm = jsn.dumps(str(list(_cm.flatten())), indent=4) 

        try:
            with open(join(path_to_file, ("%s_cm.json" %_file_name)), "w") as outfile:
                outfile.write(_cm)
        except Exception as e:
            raise e

    def runTestPrediction(self, x_test=None):
        try:
            if x_test is None:
                x_test = self.x_test

            _yp = self.cnn_model.predict(x_test)

        except Exception as e:
            raise e("prediction")
        else:
            return np.argmax(_yp, axis = 1)

        

    def saveModel(self, path_to_file):
        _file_name = ("%s_model.keras" % self.cnn_name)
        if not exists(path_to_file):
            try:
                os.makedirs(path_to_file, mode=0o740, exist_ok=True)
            except Exception as e:
                logging.error("Error: %s" % e, exc_info=True)
                raise e
        else:
            if exists(join(path_to_file, _file_name)):
                try:
                    from datetime import datetime
                    _old_file = ("%s_model_%s.h5" % (self.cnn_name, datetime.now().strftime('%m-%d-%Y_%H-%M-%S')))
                    os.rename(join(path_to_file, _file_name), join(path_to_file, _old_file))
                except Exception as e:
                    logging.error("Error: %s" % e, exc_info=True)
                    raise e

            try:
                self.cnn_model.save(join(path_to_file, _file_name), save_format='keras')
            except Exception as e:
                logging.error("Error: %s" % e, exc_info=True)
                raise e
            finally:
                logging.info("Message: SARS CNN model saved to '%s'" % join(path_to_file, _file_name), exc_info=True)



    # def loadModel(self):
    #     _home = str(Path(__file__).parents[2])
    #     _saving_path = join(_home, self.model_saving_path)
    #     _file_name = ("%s_model.h5" % self.cnn_name)

    #     if exists(_saving_path):
    #         try:
    #             self.cnn_model = tf.keras.models.load_model(join(_saving_path, _file_name))
    #             print("Message: SARS CNN model loaded from '%s'" % join(_saving_path, _file_name))
    #         except OSError as error:
    #             print("Error in CNNModelBuilder: SARS CNN model '%s' not found." % join(_saving_path, _file_name))
    #             raise error
    #     else:
    #         print("Error in CNNModelBuilder: '%s' path does not exist. Process aborted!" % join(_saving_path, _file_name))
    #         raise NotADirectoryError


    def loadDatasetAsH5pyFile(self, fileNameDatasetAsH5py):
        import h5py
        try:
            _datasetAsH5py = h5py.File(fileNameDatasetAsH5py,'r')

            _dict_group = _datasetAsH5py['classes_dict']
            _dict_group_keys = _dict_group.keys()

            for k in _dict_group_keys:
                self.classes_dict[int(k)] = _dict_group[k][()].decode()

            if self.dataset_test_divided is False:
                self.dataset_X = np.array(_datasetAsH5py['train']['X'])
                self.dataset_y = np.array(_datasetAsH5py['train']['y'])
                self.__splitDataset()
            else:
                for _sub in _datasetAsH5py.keys():
                    if _sub == 'train':
                        self.x_train = np.array(_datasetAsH5py['train']['X'])
                        self.y_train = np.array(_datasetAsH5py['train']['y'])
                    elif _sub == 'test':
                        self.x_test = np.array(_datasetAsH5py['test']['X'])
                        self.y_test = np.array(_datasetAsH5py['test']['y'])

                if self.loss_function == "categorical_crossentropy":
                    self.__datasetYInputPreprocess()


        except Exception as e:
            raise e


    def plotAccuracyGraph(self, history, fig_path_name):
        try:
            fig = plt.figure()
            plt.plot(history.history[self.training_metrics])
            plt.plot(history.history["val_%s" % self.training_metrics])
            plt.title("Model Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Trainning", "Test"], loc="upper left")
            fig.savefig("%s_accuracy.jpg" % fig_path_name)
        except Exception as e:
            raise e

    def plotLossGraph(self, history, fig_path_name):
        try:
            fig = plt.figure()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Trainning", "Test"], loc="upper left")
            fig.savefig("%s_loss.jpg" % fig_path_name)
        except Exception as e:
            raise e
    
    def __splitDataset(self):
        try:
            if self.loss_function is None:
                logging.error("Error in splitDataset: 'loss function' parameter should be not None. Process aborted!", exc_info=True)
                raise Exception

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset_X, self.dataset_y, test_size = self.test_size, random_state = 42)

            if self.loss_function == "categorical_crossentropy":
                self.__datasetYInputPreprocess()
        except Exception as e:
            raise e
        else:
            return self.x_train, self.x_test, self.y_train, self.y_test

    def set_tf_loglevel(self, level):
        # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
        # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/38645250#38645250
        import logging
        import os
        if level >= logging.FATAL:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if level >= logging.ERROR:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        if level >= logging.WARNING:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

        logging.getLogger('tensorflow').setLevel(level)