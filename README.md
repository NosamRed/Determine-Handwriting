# Determine Handwriting
A computer vision project that will detect and determine handwritten numbers and letters.

## train.py
Creates and trains the CNN model.
Saves the model as a .keras file.
Uses the EMNIST balanced datasets located in the data folder.

## app.py
Currently, tests the model by determine what character is written in the test-image.png file.
Saves a debug image called debug_preprocessed.png to show what is being passed into the CNN model.
Result is its prediction class index, its coresponding label, and its top 5 predictions along with its condience in said predictions.

## emnist_balanced_cnn.keras
The CNN model that will try to predict your character.

## test-image.png
The image that the CNN model will try to predict what character is written in said image.

## How to use
1. If a test-image.png is not provided in the root folder, create a PNG image of a clear SINGLE character ranging from 0-9 OR A-Z OR a-z named test-image.
2. If emnist_balanced_cnn.keras is already provided, simply run the app.py file to see if it predicts your chracter correctly.

## Notes:
EMNIST balanced dataset from https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip