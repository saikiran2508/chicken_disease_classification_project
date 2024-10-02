from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.prepared_callbacks import PrepareCallback
from CNN_Classifier.components.training import Training
from CNN_Classifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __int__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepared_callbacks_config = config.get_prepare_callback_config()
        prepared_callbacks = PrepareCallback(config=prepared_callbacks_config)
        callback_list = prepared_callbacks.get_tb_ckpt_callbacks()


        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(
            callback_list=callback_list
        )