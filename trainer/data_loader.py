from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
IMG_SHAPE = 20
train_dir = "../data"


def load_data():
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               validation_split=0.2,
                                               horizontal_flip=True)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               color_mode="grayscale",
                                                               target_size=(IMG_SHAPE, IMG_SHAPE),
                                                               subset="training")

    val_data_gen = train_image_generator.flow_from_directory(train_dir,
                                                             target_size=(IMG_SHAPE, IMG_SHAPE),
                                                             batch_size=BATCH_SIZE,
                                                             shuffle=False,
                                                             color_mode="grayscale",
                                                             subset="validation")

    return train_data_gen, val_data_gen
