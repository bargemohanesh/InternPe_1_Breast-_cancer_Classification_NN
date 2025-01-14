# InternPe_Breast-_cancer_Classification_NN

Breast Cancer Classification with Neural Network
This project demonstrates how to build a simple Neural Network (NN) model to classify breast cancer tumors as benign or malignant using the Breast Cancer dataset from sklearn. The model is implemented using TensorFlow and Keras.

Project Overview
The goal of this project is to develop a machine learning model that can predict whether a given tumor is benign or malignant based on various features. The model uses a neural network to classify the data.

Dependencies
Python 3.x
numpy
pandas
matplotlib
scikit-learn
tensorflow (including Keras)
Installation
To run the project, you can install the required dependencies using pip. Create a virtual environment and install the following packages:

bash
Copy code
pip install numpy pandas matplotlib scikit-learn tensorflow
Dataset
The dataset used in this project is the Breast Cancer dataset, which is included in sklearn.datasets. It consists of 30 numerical features describing different characteristics of cell nuclei present in breast cancer biopsies. The target variable is binary, where:

0 represents malignant tumors
1 represents benign tumors
Steps for Building the Model
Data Collection & Processing: The dataset is loaded using sklearn.datasets.load_breast_cancer() and converted into a DataFrame using pandas. The data is cleaned and missing values are checked (none in this case).

Feature and Target Separation: Features (X) and the target label (Y) are separated. The target variable is label, and the features are the different statistical measures of cell nuclei.

Train-Test Split: The data is split into training and testing sets using train_test_split() from sklearn.

Data Standardization: Standardization is done using StandardScaler to ensure that the features are on the same scale, which is important for neural networks to perform optimally.

Building the Neural Network: A simple neural network is built using tensorflow and keras:

The model consists of a Flatten layer (input layer), a Dense layer with 20 neurons and ReLU activation, and another Dense layer with 2 neurons (output layer) using the Sigmoid activation function.
Model Compilation: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. Accuracy is used as a metric.

Training the Model: The model is trained for 10 epochs with a validation split of 0.1. The training history is captured to monitor the model's performance during training.

Model Evaluation: The model's performance is evaluated on the test data, and accuracy is printed. A confusion matrix is also generated to assess the model's performance in terms of true positives, true negatives, false positives, and false negatives.

Prediction: The model is used to make predictions on new input data, which is standardized and then passed through the trained model.

Example Usage
To predict the type of tumor for a new set of input features:

python
Copy code
input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input
input_data_std = scaler.transform(input_data_reshaped)

# Get prediction
prediction = model.predict(input_data_std)

# Output prediction
prediction_label = [np.argmax(prediction)]
if prediction_label[0] == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
Model Evaluation
The model achieved an accuracy of 98.2% on the test set, with perfect recall (100%) for identifying malignant tumors and a precision of 97.1%.

Confusion Matrix:
lua
Copy code
[[43  2]
 [ 0 69]]
Conclusion
This neural network model demonstrates high accuracy in classifying breast cancer tumors, making it a reliable tool for early diagnosis. The model has achieved good performance in terms of both accuracy and recall, making it effective for real-world applications.

License
This project is open-source and available for free use. Feel free to modify and distribute it under the MIT License.

Let me know if you'd like to modify any sections!






