from tensorflow.keras.models import load_model
import shap
import numpy as np
from PIL import Image
import os


def load_examples():
    folder = "./visualize/examples/d_0.jpg"

    im_frame = Image.open(folder)
    np_frame = np.array(im_frame.getdata())
    return np_frame


def visualize():
    background = load_examples()
    model = load_model("./models/classifier.h5")
    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)


if __name__ == "__main__":
    visualize()
