import os
import numpy as np
import random
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import sqlite3
import json, dbutils
#TODO
#ukladani do db pro vyhledavani
imgs_per_class=10000


def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
def img_to_np(path,x=224,y=224):
    nparray = image.load_img(path,target_size=(x,y))
    nparray = image.img_to_array(nparray)
    nparray = np.expand_dims(nparray, axis=0)
    nparray = preprocess_input(nparray)
    nparray = np.squeeze(nparray)
    return nparray
class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self,model,graph, dim_x = 224, dim_y = 224, batch_size = 10, dataset_path = './places365-dataset/20_classes'):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.model=model
        self.graph=graph
        self.conn=sqlite3.connect('representations.db')
        dbutils.create()


    def generate(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            #indexes = self.__get_exploration_order(list_IDs)
            image_IDs = self.__make_triplets()

            # Generate batches
            imax = int(len(image_IDs)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                #list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                image_IDS_temp = image_IDs[i*self.batch_size:(i+1)*self.batch_size]

                # Generate data
                #X, y = self.__data_generation(labels, list_IDs_temp)
                X = self.__data_generation(image_IDS_temp)
                #y_anch = np.ones((self.batch_size, 1)) # not used by triple loss function
                #y_pos = np.ones((self.batch_size, 1)) # not used by triple loss function
                #y_neg = np.ones((self.batch_size, 1)) # not used by triple loss function
                y_stacked = np.ones((self.batch_size,2, 1)) # not used by triple loss function
                yield X,y_stacked#,y_anch,y_pos,y_neg]

    def __get_subdirectories(self, a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def _img_to_np(self,path):
        nparray = image.load_img(os.path.join(self.dataset_path, path ),
                                  target_size=(self.dim_y, self.dim_x))
        nparray = image.img_to_array(nparray)
        nparray = np.expand_dims(nparray, axis=0)
        nparray = preprocess_input(nparray)
        nparray = np.squeeze(nparray)
        return nparray
    def __make_triplets(self):
        
        np.random.seed(0)

        classes = self.__get_subdirectories(self.dataset_path)

        all_triplets = []
        for Id, c in enumerate(classes):
            print ("generating triplets for %s (%s/%s)"%(c,Id+1,len(classes)))
            pos_dir = os.path.join(self.dataset_path, c)
            imgs_pos = os.listdir(pos_dir)
            class_triplets = []
            anchor_batch = []
            positive_batch = []
            negative_batch = []
            anchors=[]
            positives=[]
            negatives=[]
            for idx in range(0,len(imgs_pos),2):
                if not idx%500:
                    print ("%s/%s"%(idx,min(len(imgs_pos),imgs_per_class)))
                if idx>=imgs_per_class:
                    break
                anchor=img_to_np(os.path.join(self.dataset_path, c + '/' + imgs_pos[idx]))
                try:
                    positive=img_to_np(os.path.join(self.dataset_path, c + '/' + imgs_pos[idx+1]))
                except IndexError:
                    idx=-1
                    positive=img_to_np(os.path.join(self.dataset_path, c + '/' + imgs_pos[idx+1]))

                rand_class = random.choice([x for x in classes if x != c])  # choose a different class randomly
                #print ("Positive: %s"%c)
                #print ("rand_class: %s"%rand_class)
                neg = random.choice(os.listdir(os.path.join(self.dataset_path, rand_class)))
                negative1 = img_to_np(os.path.join(self.dataset_path, rand_class + '/' + neg))

                #rand_class = random.choice([x for x in classes if x != c])  # choose a different class randomly
                #neg = random.choice(os.listdir(os.path.join(self.dataset_path, rand_class)))
                #negative2 = self._img_to_np(rand_class + '/' + neg)



                anchor_batch.append(anchor)
                anchors.append(imgs_pos[idx])
                positive_batch.append(positive)
                positives.append(imgs_pos[idx+1])
                negative_batch.append(negative1)
                negatives.append(rand_class + '/'+ neg)
                #negative_batch.append(negative2)
            print ("Calculating representations for %s"%c)
            with self.graph.as_default():
                preds = self.model.predict(
                    [np.asarray(anchor_batch), np.asarray(positive_batch), np.asarray(negative_batch)])
            # print (preds)
            # print (preds.shape)
            #preds_pos = np.concatenate(np.asarray(preds[1]),np.asarray(preds[2]))
            preds_anch = np.asarray(preds[0])
            preds_pos = np.asarray(preds[1])

            preds_neg = np.asarray(preds[2])
            #print (preds_pos)
            #print (preds_neg)
            #print (preds_anch.shape)
            #print (preds_pos.shape)
            #print (preds_neg.shape)
            print ("Calculating distances for %s"%c)

            #anch_pairs=[("%s/%s"%(c,matrix[0]),matrix[1]) for matrix in zip(imgs_pos,preds_anch)]
            #pos_pairs=[("%s/%s"%(c,matrix[0]),matrix[1]) for matrix in zip(positives,preds_pos)]
            #print (anch_pairs)
            conn = sqlite3.connect('representations.db')
            cur = conn.cursor()

            for name, matrix in zip(anchors,preds_anch):
                name="%s/%s"%(c,name)
                cur.execute("REPLACE INTO images VALUES (?,?,?)", (None,name, json.dumps(matrix.tolist())))
            for name, matrix in zip(positives,preds_pos):
                name="%s/%s"%(c,name)
                cur.execute("REPLACE INTO images VALUES (?,?,?)", (None,name, json.dumps(matrix.tolist())))


            conn.commit()

            for i,anch in enumerate(preds_anch):
                least_sim_pos_idx,most_sim_neg_idx=self._least_similar(preds_anch[i],preds_pos,preds_neg)
                """ Takes two images from the same class/folder and one image from the next class/folder """
                #print(imgs_pos[i])
                class_triplets.append([c+'/'+imgs_pos[i], c+'/'+positives[least_sim_pos_idx], negatives[most_sim_neg_idx]])

            all_triplets += class_triplets

        triplets = np.array(all_triplets)
        np.save("triplets.npy",triplets)
        np.random.shuffle(triplets)
        #print (triplets)
        #print (triplets.shape)

        return triplets




    # get the least similar image from the same class
    def _least_similar(self, anch, preds_pos, preds_neg):
        def euclidean_distance(x,y):
            return np.linalg.norm(x-y)
        least_sim_pos=preds_pos[0]
        least_sim_pos_idx=0
        least_sim_dist=euclidean_distance(anch,least_sim_pos)
        for i,candidate in enumerate(preds_pos):
            if euclidean_distance(anch,candidate)>least_sim_dist:
                least_sim_pos = candidate
                least_sim_pos_idx=i
                least_sim_dist = euclidean_distance(anch, least_sim_pos)

        most_sim_neg = preds_pos[0]
        most_sim_neg_idx = 0

        most_dist = euclidean_distance(anch, most_sim_neg)
        for i,candidate in enumerate(preds_neg):
            if euclidean_distance(anch, candidate) < most_dist:
                most_sim_neg = candidate
                most_sim_neg_idx=i
                most_dist = euclidean_distance(anch, most_sim_neg)

        #print (least_sim_dist,most_dist)
        return least_sim_pos_idx,most_sim_neg_idx
#        print (preds)
 #       print ("______________________________________")
        #print ((image_anch,filtered_pos[np.argmax(preds_pos,axis=0)]))
        #return filtered_pos[np.argmax(preds,axis=0)]

    def __data_generation(self, image_IDs):

        anchor_batch = []
        positive_batch = []
        negative_batch = []

        for img_path in image_IDs:
            #print (img_path)
            anchor = img_to_np(os.path.join(self.dataset_path, img_path[0]))
            positive =  img_to_np(os.path.join(self.dataset_path, img_path[1]))
            negative =  img_to_np(os.path.join(self.dataset_path, img_path[2]))

            anchor_batch.append(anchor)
            positive_batch.append(positive)
            negative_batch.append(negative)

        return [np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)]


