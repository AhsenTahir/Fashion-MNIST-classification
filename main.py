import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Loading the dataset
(Image_train, Label_train), (Image_test, Label_test) = fashion_mnist.load_data()

# Normalizing the images
Image_train,Image_test=Image_train/255.0,Image_test/255.0

# Building the model
model = Sequential([
 Flatten(input_shape=[28, 28]),
 Dense(300, activation="relu"),
 Dense(100, activation="relu"),
 Dense(10, activation="softmax")
])

# Compiling the model
model.compile(optimizer=Adam(),loss=SparseCategoricalCrossentropy(),metrics=["accuracy"])

# Training the model
history=model.fit(Image_train, Label_train, epochs=10, batch_size=32, validation_split=0.2)
#evaluating the model
test_loss, test_accuracy = model.evaluate( Image_test, Label_test)
print(f"Test accuracy: {test_accuracy}")

# Making predictions on 5th image
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

predictions = model.predict(np.expand_dims(Image_test[4], axis=0))
predicted_class_idx = np.argmax(predictions)
predicted_class_name = class_names[predicted_class_idx]
print(f"Model prediction: {predicted_class_name}")

# Comparing with actual label
actual_label_idx = Label_test[4]
actual_class_name = class_names[actual_label_idx]
print(f"Actual class: {actual_class_name}")