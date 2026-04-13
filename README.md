# Determine Handwriting
A computer vision project that will detect and determine handwritten numbers and letters.

## `web.py`
* This is where you will interact with the tool via a locally hosted webside.
* Two options of prediciton:
    1. Uploading a image:
        * You can upload a clear image of a character for the model to try to predict.
    2. Drawing an character:
        * You can draw the character for the model to predict.

## `train.py`
* Creates and trains the CNN model.
* Saves the model as a `.keras` file.
* Uses the EMNIST balanced datasets located in the `training-data` folder.

## `predictor.py`
* Looks at a a provided character from `web.py` to predict what the character is.
* Results returned is its prediction class index, its coresponding label, and its top 5 predictions along with its condience in said predictions.

## `emnist_balanced_cnn.keras`
* The CNN model that will try to predict your character.

## Notes:
* EMNIST balanced dataset from https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip