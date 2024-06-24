# Fashion MNIST Classification with TensorFlow

This project utilizes TensorFlow to create a deep learning model for classifying fashion items from the Fashion MNIST dataset.

### Overview:

- **Dataset:** Fashion MNIST dataset is used, containing 60,000 training images and 10,000 test images of 10 different fashion categories.
- **Model Architecture:** Sequential neural network with three dense layers:
  - Input layer: Flatten layer to reshape images.
  - Hidden layers: Two dense layers with ReLU activation (300 neurons and 100 neurons).
  - Output layer: Dense layer with softmax activation (10 neurons for 10 classes).
- **Training:** The model is trained using Adam optimizer, SparseCategoricalCrossentropy loss function, and evaluates accuracy metrics.
- **Prediction:** After training, the model predicts the class of a given test image and compares it with the actual label.

### Usage:

1. **Environment Setup:**
   - Ensure TensorFlow and required dependencies are installed (`pip install -r requirements.txt`).

2. **Running the Code:**
   - Execute the Python script (`python main.py`) to train the model, evaluate accuracy, and make predictions.

3. **Understanding Results:**
   - Check console output for test accuracy and individual predictions.
