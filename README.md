# Real-time Crime Detection

This GitHub repository contains a project that aims to develop a real-time crime detection model using computer vision and deep learning techniques. The model is trained using an LSTM-CNN architecture and deployed as a Flask web application.

## Directory Structure

The repository has the following directory structure:

- `Models`: This directory contains the trained crime detection model.
- `templates`: This directory contains HTML files for the web application.
- `procfile`: This file specifies the commands that are executed by the app on Heroku.
- `requirements.txt`: This file lists the dependencies required to run the Flask application.
- `app.py`: This file contains the main code for running the Flask web application.

## Project Overview

The goal of this project is to develop a real-time crime detection system that utilizes computer vision and deep learning algorithms. The model in this project is trained using an LSTM-CNN architecture, which combines the power of both LSTM (Long Short-Term Memory) and CNN (Convolutional Neural Network) models to detect and classify criminal activities from video footage.

The Flask web application allows users to interact with the crime detection model in real-time. Users can act infront of camera to capture live video which is fed to the application, and will process the frames using the trained model and provide real-time crime detection results. The HTML templates in the `templates` directory define the structure and layout of the web pages.

## Getting Started

To get started with this project, please follow these steps:

1. Clone the repository: `git clone https://github.com/rashant/Real-time-crime-detection.git`
2. Install the required dependencies by running: `pip install -r requirements.txt`
3. Ensure that you have Python and Flask installed on your system.
4. Run the Flask web application using the command: `python app.py`
5. Access the application by opening a web browser and navigating to `http://localhost:5000`.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code as per the terms of the license.
