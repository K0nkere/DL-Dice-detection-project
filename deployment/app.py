# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import numpy as np

from PIL import Image
from io import BytesIO
import zlib
from urllib import request as urlrequest

from sklearn.cluster import DBSCAN

from flask import Flask, request, jsonify, Response

img_size=128

detection_model = "models/detection-model-dr03-0729.tflite"
viz_model = "models/viz-model-dr03-0729.tflite"

classes = [
    'd10',
    'd12',
    'd20',
    'd4',
    'd6',
    'd8',
    'dicesback'
]

def download_image(url):
    
    with urlrequest.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(X, rescale=255):
    
    if rescale:
        X = X*1./rescale
        return X

    return X

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


def get_predictions(X, model_path="detection-model-dr03-0729.tflite"):
    
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, X.astype("float32"))
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_index)[0]

    return predictions

def get_mask(features, threshold=1.5): #sample_url, viz_model, n_layer):
    """
    Mask selection based on mean feature maps approach
    """
    
    num_filters = features.shape[-1]
    
    avg_map = np.mean(features)

    map_mask = [1 if avg_map<np.mean(features[:, :, i]) else 0 for i in range(num_filters)]
    
    mean_mask = np.mean(features[:,:, map_mask], axis=-1)
    
    mean_mask[mean_mask < threshold*np.mean(mean_mask)] = 0
    
    return mean_mask

def get_clusters(mask, n_map = 0):
    
    get_labels = DBSCAN(eps=5, min_samples=35)

    points = list(zip(*np.where(mask[:,:,n_map] >0)))

    if points:
        labels = get_labels.fit_predict(points)
        
        labeled_mask = np.zeros((img_size, img_size, len(np.unique(labels))))
        
        for i, j, l in zip(*(np.where(mask[:,:,n_map] >0)), labels):
            if l>=0:
                labeled_mask[i, j, l] = 1

        if np.sum(labeled_mask[:,:, -1]) == 0:
            labeled_mask = labeled_mask[:, :, :-1]

        return labeled_mask

    else:
        print("No dices detected")
        
        return None


def get_anchors(labeled_masks):
    clusters = labeled_masks.shape[2]

    anchors = []
    for cluster in range(clusters):
        yx_locs = np.where(labeled_masks[:,:, cluster])

        x_center = int(np.mean(yx_locs[1]))
        y_center = int(np.mean(yx_locs[0]))

        dx = np.ceil((np.max(yx_locs[1]) - np.min(yx_locs[1]))/2.)
        dy = np.ceil((np.max(yx_locs[0]) - np.min(yx_locs[0]))/2.)

        if dx < dy:
            dh = int(dy)
        else:
            dh = int(dx)

        anchors.append([x_center, y_center, dh])
    
    return anchors


def plot_boxes(boxed_sample, anchors, dh=16):
    
    size = boxed_sample.shape[0]
    
    for anchor in anchors:

        x_center = anchor[0]
        y_center = anchor[1]
        
        h = max(anchor[2], dh)

        x_min = max(0, x_center - h)
        x_max = min(size-1, x_center + h)
        y_min = max(0, y_center - h)
        y_max = min(size-1, y_center + h)


        box_mask_hmin = [[1 if (i>=x_min)&(i<=x_max)&(j==y_min) else 0 for i in range(size)] for j in range(size)]
        box_mask_hmax = [[1 if (i>=x_min)&(i<=x_max)&(j==y_max) else 0 for i in range(size)] for j in range(size)]
        box_mask_vmin = [[1 if (i==x_min)&(j>=y_min)&(j<=y_max) else 0 for i in range(size)] for j in range(size)]
        box_mask_vmax = [[1 if (i==x_max)&(j>=y_min)&(j<=y_max) else 0 for i in range(size)] for j in range(size)]

        box_mask = np.sum([box_mask_hmin, box_mask_hmax, box_mask_vmin, box_mask_vmax], axis=0)


        boxed_sample[:,:, 0][np.where(box_mask == 1)] = 1
        boxed_sample[:,:, 1][np.where(box_mask == 1)] = 0
        boxed_sample[:,:, 2][np.where(box_mask == 1)] = 0
    
    return boxed_sample


def predict(sample):
    
    x = np.array(sample)
    x = np.array([x])
    X = preprocess(x, 255)

    f_maps = get_predictions(X, model_path=viz_model)

    f_mask = get_mask(f_maps, threshold=1.6)

    mask = f_mask[:, :, np.newaxis].copy()

    labeled_masks = get_clusters(mask)
    anchors = get_anchors(labeled_masks=labeled_masks)

    if anchors:

        valid_anchors = []
        for anchor in anchors:

            x_center = anchor[0]
            y_center = anchor[1]
            h = anchor[2] + 5 

            x_min = max(0, x_center - h)
            x_max = min(img_size-1, x_center + h)
            y_min = max(0, y_center - h)
            y_max = min(img_size-1, y_center + h)

            img_sample = sample.resize((img_size, img_size), box=(x_min, y_min, x_max, y_max))

            x_sample = np.array(img_sample)
            x_sample = np.array([x_sample])
            X_sample = preprocess(x_sample)

            classes_prob = get_predictions(X_sample, detection_model)

            result = sorted(dict(zip(classes, classes_prob)).items(), reverse=True, key=lambda x: x[1])[0]
            

            if result[0] != 'dicesback':
                print(f"Dice probability: {result[1]:.2f}")
                valid_anchors.append(anchor)
            else:
                print(f"Cluster classified as background: {result[1]:.2f}")

        boxed_sample = plot_boxes(X[0], valid_anchors, dh=15)

        return boxed_sample

    else:
        
        return None


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


    sample_img_original = download_image(sample_url)
    sample = prepare_image(sample_img_original, (img_size, img_size))
    
    result = predict(sample)
    
    if not isinstance(result, np.ndarray):

        return jsonify("No dices detected")

    resp, _, _ = compress_nparr(result)
    
    return Response(response=resp, status=200, mimetype="application/octet_stream")


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, port=9696)
