from json import load
import logging
import sys
import numpy as np
import tensorflow


def init():
    # load model
    loaded_model = tensorflow.keras.models.load_model('model.h5')
    logging.info("Successfully initialized keras model")
    loaded_model.summary()
    graph = tensorflow.Graph()
    return loaded_model,graph

