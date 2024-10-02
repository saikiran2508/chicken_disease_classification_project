from CNN_Classifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
import numpy as np

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):

        # datagenerator_kwargs = dict(
        #     rescale = 1./255,
        #     validation_split=0.20
        # )

        # dataflow_kwargs = dict(
        #     target_size=self.config.params_image_size[:-1],
        #     batch_size=self.config.params_batch_size,
        #     interpolation="bilinear"
        # )

        def generator_to_dataset(generator):
            output_signature = (
                tf.TensorSpec(shape=(None, *generator.image_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(None, generator.num_classes), dtype=tf.float32)
            )
            dataset = tf.data.Dataset.from_generator(
                lambda: generator,
                output_signature=output_signature
            )
            return dataset


        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
            validation_split=0.20
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        self.valid_dataset = generator_to_dataset(self.valid_generator).repeat()

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                rescale = 1./255,
                validation_split=0.20
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        self.train_dataset = generator_to_dataset(self.train_generator).repeat()

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):

        self.steps_per_epoch = int(np.ceil(self.train_generator.samples / self.train_generator.batch_size))
        self.validation_steps = int(np.ceil(self.valid_generator.samples / self.valid_generator.batch_size))

        self.model.fit(
            self.train_dataset,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_dataset,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.training_model_path,
            model=self.model
        )