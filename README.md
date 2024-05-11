
# Pneumonia Detection using CNN

## Overview

This project aims to detect pneumonia in chest X-ray images using Convolutional Neural Networks (CNN). It includes a Jupyter notebook for model training and evaluation and a Flask web application for real-time inference on user-uploaded images.

## Requirements

- Python 3.x
- TensorFlow
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Flask

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/pneumonia-detection-using-cnn.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Jupyter Notebook (Model Training and Evaluation)

1. Load the dataset: Ensure you have downloaded and placed the dataset appropriately. You can modify the paths in the code if necessary.

2. Execute the notebook: Run the provided Jupyter notebook `pneumonia_detection.ipynb` to go through the data preprocessing, model building, training, and evaluation steps.

3. Adjust parameters: You can tweak the model architecture, data augmentation techniques, and hyperparameters to improve performance.

4. Evaluate the model: After training, evaluate the model's performance on the test set and analyze the results.


### Flask Web Application (Real-time Inference)

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Visit `http://localhost:5000` in your web browser to use the web application.

## Web Application

The Flask web application allows users to upload chest X-ray images and receive real-time predictions on whether pneumonia is detected.

- `/`: Home page with an upload form to submit images for prediction.
- `/about`: About page providing information about the project.

## Model

The trained model (`my_model3.h5`) is loaded into the Flask application for inference. It uses a pre-trained VGG16 architecture fine-tuned for pneumonia detection.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data).
- We acknowledge the developers of TensorFlow, scikit-learn, Flask, and other open-source libraries used in this project.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
