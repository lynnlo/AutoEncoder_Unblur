# Imports

# System
import os
import sys

# Processing
import numpy as np
from PIL import Image
from matplotlib import pyplot

# Keras
import tensorflow
from tensorflow import keras
from keras import Model, Input
from keras.preprocessing.image import img_to_array, array_to_img
from keras.layers import Dense, Flatten

tensorflow.executing_eagerly()

# Preprocess
if not os.path.exists("data.npy"):
	images = ["./archive/"+i for i in os.listdir('./archive')][0:64]

	allimages = []
	allarrays = []

	print("filtering")
	for i in images:
		img = Image.open(i).resize((64, 64)).convert("L")
		ary = img_to_array(img, dtype=float).reshape(64 * 64)
		allimages.append(img)
		allarrays.append(ary)

	allarrays = np.array(allarrays).reshape(-1, 64 * 64)
	print("normalizing")
	allarrays /= 255

	testimage = Image.open(images[1]).resize((64, 64)).convert("L")

	testarray = img_to_array(testimage, dtype=float).reshape(1, 64 * 64)

	print("saving")
	np.save("data.npy", allarrays)
else:
	print("loading data")
	allarrays = np.load("data.npy")
	testimage = Image.open(["./archive/"+i for i in os.listdir('./archive')][1]).resize((64, 64)).convert("L")
	testarray = img_to_array(testimage, dtype=float).reshape(1, 64 * 64)

# Model
if not os.path.exists("./models/model_test_1.h5"):
	print("compiling")
	x_size = 64
	y_size = 64

	size = x_size * y_size
	layer1_size = round(size * 0.9)
	layer2_size = round(size * 0.8)

	img_input = Input(shape=(size,))

	encoder1 = Dense(layer1_size, activation="sigmoid")(img_input)
	encoder2 = Dense(layer2_size, activation="sigmoid")(encoder1)

	decoder1 = Dense(layer1_size, activation="sigmoid")(encoder2)
	decoder2 = Dense(size, activation="sigmoid")(decoder1) 

	autoencoder = Model(inputs=img_input, outputs=decoder2)
	autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
else:
	print("loading model")
	autoencoder = keras.models.load_model("./models/model_test_1.h5")

print("training")
autoencoder.fit(x=allarrays, y=allarrays, batch_size=64, epochs=128)

print("saving")
autoencoder.save("./models/model_test_1.h5")

print("showing")
pyplot.figure()
pyplot.imshow(testimage, cmap="gray")
pyplot.figure()
pyplot.imshow(array_to_img(autoencoder.predict(testarray)[0].reshape(64, 64, 1)), cmap="gray")
pyplot.show()