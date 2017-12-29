import os
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, dim_x = 224, dim_y = 224, batch_size = 10, dataset_path = './places365-dataset/20_classes'):
		'Initialization'
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.batch_size = batch_size
		self.dataset_path = dataset_path

	def generate(self):
		'Generates batches of samples'
		# Infinite loop
		while 1:
			# Generate order of exploration of dataset
			image_IDs = self.__make_triplets()
			print('num. of triplets: ', image_IDs.shape)

			# Generate batches
			imax = int(len(image_IDs)/self.batch_size)
			for i in range(imax):
				# Find list of IDs
				image_IDS_temp = image_IDs[i*self.batch_size:(i+1)*self.batch_size]

				# Generate data
				X = self.__data_generation(image_IDS_temp)
				y = np.ones((self.batch_size, 2, 1)) # not used by triple loss function

				yield X, y

	def __get_subdirectories(self, a_dir):
		return [name for name in os.listdir(a_dir)
				if os.path.isdir(os.path.join(a_dir, name))]

	def __make_triplets(self):

		classes = self.__get_subdirectories(self.dataset_path)

		np.random.seed(0)
		classes_num = len(classes)

		all_triplets = []
		
		for Id, c in enumerate(classes):

			next_class_id = Id
			while (next_class_id == Id):
				next_class_id = np.random.randint(classes_num, size=1)[0]

			pos_dir = os.path.join(self.dataset_path, c)
			neg_dir = os.path.join(self.dataset_path, classes[next_class_id])

			imgs_pos = os.listdir(pos_dir)
			imgs_neg = os.listdir(neg_dir)

			class_triplets = []
			for i, img in enumerate(imgs_pos):
				""" Takes two images from the same class/folder and one image from the next class/folder """
				class_triplets.append((c+'/'+imgs_pos[i], c+'/'+imgs_pos[(i+1) % len(imgs_pos)], classes[next_class_id]+'/'+imgs_neg[i%len(imgs_neg)]))

			all_triplets += class_triplets

		triplets = np.array(all_triplets)
		np.random.shuffle(triplets)

		return triplets

	def __data_generation(self, image_IDs):

		anchor_batch = []
		positive_batch = []
		negative_batch = []

		for img_path in image_IDs:
			anchor = image.load_img(os.path.join(self.dataset_path, img_path[0]), target_size=(self.dim_y, self.dim_x))
			anchor = image.img_to_array(anchor)
			anchor = np.expand_dims(anchor, axis=0)
			anchor = preprocess_input(anchor)
			anchor = np.squeeze(anchor)

			positive = image.load_img(os.path.join(self.dataset_path, img_path[1]), target_size=(self.dim_y, self.dim_x))
			positive = image.img_to_array(positive)
			positive = np.expand_dims(positive, axis=0)
			positive = preprocess_input(positive)
			positive = np.squeeze(positive)

			negative = image.load_img(os.path.join(self.dataset_path, img_path[2]), target_size=(self.dim_y, self.dim_x))
			negative = image.img_to_array(negative)
			negative = np.expand_dims(negative, axis=0)
			negative = preprocess_input(negative)
			negative = np.squeeze(negative)


			anchor_batch.append(anchor)
			positive_batch.append(positive)
			negative_batch.append(negative)

		return [np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)]
