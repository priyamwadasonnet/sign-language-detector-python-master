import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pickle
import os
import pandas as pd

# Load the trained Random Forest model and label encoder
try:
    model_data = pickle.load(open('model.p', 'rb'))
    rf_model = model_data['model']
    label_encoder = model_data['label_encoder']
    print("Model and label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: Model or label encoder file not found.")
    exit()

# Prepare TensorFlow Dataset directly from numpy arrays (bypassing pandas)
def create_tf_dataset(data, labels):
    """
    Converts numpy arrays to a TensorFlow Decision Forests Dataset.
    """
    # Assuming data is a numpy array and labels are integer encoded
    features = {f"feature_{i}": tf.convert_to_tensor(data[:, i], dtype=tf.float32) for i in range(data.shape[1])}
    features['label'] = tf.convert_to_tensor(labels, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices(features)

# Load data for re-export
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data = data_dict['data']
    labels = label_encoder.transform(data_dict['labels'])  # Ensure labels are properly encoded
    print("Data and labels loaded and encoded successfully.")
except FileNotFoundError:
    print("Error: Data file not found.")
    exit()

# Convert to TensorFlow dataset
dataset = create_tf_dataset(data, labels)

# Configure and train TensorFlow Decision Forest model
tfdf_model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION, 
                                           num_trees=100,  # Optionally, tweak number of trees
                                           max_depth=10)   # Set a reasonable depth for the trees

try:
    tfdf_model.fit(dataset)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error during model fitting: {e}")
    exit()

# Save TensorFlow model with better handling of feature names and metadata
try:
    tfdf_model.save('gesture_model_tf')
    print("Model saved successfully as 'gesture_model_tf'.")
except Exception as e:
    print(f"Error while saving model: {e}")
    exit()
