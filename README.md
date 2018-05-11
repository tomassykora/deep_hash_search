# deep_hash_search
Image search based on convolutional neural network feature extraction.

## Instructions

Download the dataset and extract it:

	wget -c http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
	tar -xf train_256_places365standard.tar

Run to split the dataset into training and testing parts:

	./train_test_split.sh

### Training:

To train with hard mining (web api available):

	launch training with
		python3 resnet.py

After the model is trained and saved, you can run the sample website: ```FLASK_APP=web_pova.py flask run```

To train without hard mining (web api not available):

	launch training with
		python3 model_noHardMin.py

To evaluate the saved trained model run:

	python3 eval_results.py

Requirements: Keras, tensorflow, [Flask - for web api]

Authors: 
 - Tomáš Sýkora (tms.sykora@gmail.com)
 - Josef Jon (xjonjo00@stud.fit.vutbr.cz)
