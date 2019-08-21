import tensorflow as tf

import logging

from trainer.data_loader import load_data
from trainer.model import create_model
from visualize.plot import plot_acc

BATCH_SIZE = 32
EPOCHS = 20


def train_and_evaluate():
    train_data_gen, val_data_gen = load_data()
    model = create_model()

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=(train_data_gen.samples // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=(val_data_gen.samples // BATCH_SIZE)
    )

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plot_acc(EPOCHS, acc, val_acc)


if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    print(tf.__version__)

    train_and_evaluate()
