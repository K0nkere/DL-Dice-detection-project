# run under the conda environment

import tensorflow as tf
from tensorflow import keras

n_layer = 4
img_size = 128

# detection_model = keras.models.load_model("models/dice-detection-model-dr03-0.729.h5")
detection_model = keras.models.load_model("models/dice-detection-model-std-lanc-dr03-0.790.h5")

# xception_classifier = keras.models.load_model("models/xception-classifier-prepr-dr075-0.980.h5")
xception_classifier = keras.models.load_model("models/xception-classifier-prepr-lancoz-dr075-0.983.h5")

inputs = keras.Input(shape=(img_size, img_size, 3))

viz_model = keras.Model(inputs=detection_model.inputs, outputs=detection_model.layers[n_layer].output)

converter = tf.lite.TFLiteConverter.from_keras_model(viz_model)
tf_lite_model = converter.convert()

with open('models/viz-model.tflite', 'wb') as f_out:
    f_out.write(tf_lite_model)

xception_converter = tf.lite.TFLiteConverter.from_keras_model(xception_classifier)
tf_lite_model = xception_converter.convert()

with open('models/xception-classifier.tflite', 'wb') as f_out:
    f_out.write(tf_lite_model)