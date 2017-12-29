import  evaluate
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import sqlite3, json, os
from itertools import islice
from resnet import fake_loss
import resnet
import keras.losses
from keras.applications.resnet50 import ResNet50

keras.losses.fake_loss = fake_loss
def search( img_path, model_file="model.h5"):
    #with open("model.json") as f:
    #    model = model_from_json(f.read())
    #model.load_weights(model_file)
    model=load_model(model_file)
    batch=[]
    img = image.load_img(os.path.join(img_path), target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img = np.squeeze(img)
    batch.append(img)
    preds=model.predict(np.asarray(batch))

    conn = sqlite3.connect('representations.db')
    cur = conn.cursor()
    files=[]
    rep=[]
    for row in cur.execute('SELECT * FROM images'):
        files.append(row[1])
        rep.append(np.asarray(json.loads(row[2]),dtype="float32"))
    return evaluate.sim_sort(preds,files,rep)
if __name__=="__main__":
    resnet.update_db(load_model("model.h5"),dataset="static/data_train")
    search("static/data_train/airfield/00000450.jpg")
