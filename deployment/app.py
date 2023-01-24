from flask import Flask, request, jsonify, Response

import numpy as np
from io import BytesIO
import zlib

import main
from main import Detector

img_size=128
classes = ['d10','d12','d20','d4','d6','d8','dicesback']

prefix = "../"
detection_model = f"{prefix}models/xception-classifier-prepr-dr075-0.980.tflite"
viz_model = f"{prefix}models/viz-model-dr03-0729.tflite"

preprocess_type = "xception"

predictor = Detector(img_size=128, preprocess_type=preprocess_type, classes=classes, detection_model=detection_model, viz_model=viz_model)


def compress_nparr(nparr):
    """
    code from https://gist.github.com/andres-fr/f9c0d5993d7e7b36e838744291c26dde
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = BytesIO()
    np.save(bytestream, nparr)

    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)

    return compressed, len(uncompressed), len(compressed)


app = Flask("dnd-dice-detection")


@app.route("/predict", methods=["POST"])
def handler():
    """
    """
    
    # sample_url = event['url']
    event = request.get_json()
    # print(event)
    sample_url = event['url']
    # print(sample_url)


    sample_img_original = main.download_image(sample_url)
    sample = main.prepare_image(sample_img_original, (img_size, img_size))
    
    
    result = predictor.predict(sample)
    
    if not isinstance(result, np.ndarray):

        return jsonify("No dices detected")

    resp, _, _ = compress_nparr(result)
    
    return Response(response=resp, status=200, mimetype="application/octet_stream")


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, port=9696)
