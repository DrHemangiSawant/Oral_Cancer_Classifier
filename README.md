# Oral Cancer Classifier - Detecting Oral Cancer, One Convolution at a Time

# About Dataset:
This dataset contains histopathologic images of oral tissue samples, categorized as cancerous and non-cancerous. The objective is to predict the presence of oral cancer based on pixel-level features extracted from these images. The dataset serves as the foundation for training machine learning models to identify oral cancer through image classification.

# Content of Dataset:
The dataset consists of histopathologic images and one target variable, Cancerous. The image data is organized into two categories:

Cancerous: Tissue samples showing signs of oral cancer.
Non-Cancerous: Healthy tissue samples.
The dataset also includes metadata on the images, such as resolution and tissue type, which may provide additional context for further analysis.

# About Analysis:
Histopathologic Image Analysis: The images were processed and classified based on pixel-level features to detect the presence of oral cancer. Convolutional Neural Networks (CNN) were used to extract patterns from the pixel data for classification.

Training & Testing: The dataset was split into training, testing, and validation sets. The model was trained on the training set, validated on the validation set, and tested on the test set to assess generalization performance.

Pixel-Level Analysis: The model analyzes pixel-level features to differentiate cancerous from non-cancerous tissue. The CNN architecture extracts meaningful features automatically from the images.

Model Evaluation: The model's performance was evaluated using Accuracy and Loss metrics over the epochs. These metrics track the model's learning process during training and provide insights into its performance.

# Visualizations:
The following plots were generated during the analysis:

Accuracy over Epochs: Shows how the model's accuracy improved as it learned over time.
Loss over Epochs: Displays the loss reduction during training, indicating the model's progress toward minimizing errors.

# Tools and Libraries Used:
Python for scripting.
TensorFlow / Keras for building and training the CNN model.
OpenCV for image preprocessing.
Matplotlib for creating visualizations.
NumPy for numerical operations.
Pandas for data handling and manipulation.
Jupyter Notebook for interactive code development.

# Conclusion:
This analysis demonstrates how Convolutional Neural Networks (CNNs) can be applied to histopathologic images for the detection of oral cancer. By analyzing pixel-level features, the model is able to accurately classify cancerous tissue. The visualizations, including accuracy and loss plots over epochs, offer a clear view of the model's training process and its performance.
