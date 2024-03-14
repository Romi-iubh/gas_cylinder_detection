# Industrial Gas Cylinder Detection Model
## Description
This repository contains a demonstration of the Mask R-CNN model developed during my master's thesis for detecting industrial gas cylinders. The model was trained using a dataset composed of 90% synthetic images and 10% real ones.

To showcase the model's performance, we have included 10 real test images sourced from the internet, which differ from the test images used in the thesis due to confidentiality reasons.

Feel free to experiment with the model using additional online images. Suggested search terms include "industrial gas cylinder nitrogen" or "industrial gas cylinder argon". However, please note that the model was specifically optimized for:
- Industrial gas cylinders equipped with shrouds
- Pictures taken from a viewpoint higher than the cylinders
- "Non-LNG-shape" gas cylinders (containing nitrogen, oxygen, argon, hydrogen, etc.)

The model may struggle with images depicting cylinders without shrouds (a poor safety practice) or from lower viewpoints.

## Contents
This repository includes:

- The trained Mask R-CNN model
- Samples of various types of synthetic datasets
- A demo notebook to test the model with the provided real images or your own images

