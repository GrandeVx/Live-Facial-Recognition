import os
import gdown
from model.Facenet import InceptionResNetV2

def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
    weights_path=None,
):

    model = InceptionResNetV2(dimension=512)

    # -------------------------

    if os.path.isfile(weights_path) != True:
        print("facenet512_weights.h5 will be downloaded...")
        
        output = "./weights/facenet512_weights.h5"
        os.makedirs(os.path.dirname(output), exist_ok=True)

        gdown.download(url, output, quiet=False)
        weights_path = output

    # -------------------------

    model.load_weights(weights_path)

    # -------------------------

    return model




