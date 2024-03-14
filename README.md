# Industrial Gas Cylinder Detection Model

## Description
This repository contains a demonstration of the Mask R-CNN model developed during my master's thesis for detecting industrial gas cylinders.  The model was trained using 90% of synthetic images and 10% of real ones.

To showcase the model's performance, 10 test real images have been sourced from the internet. Please note that these images differ from the test images used in the thesis work due to confidentiality reasons.

You are encouraged to experiment with the model using additional images found online. Suggested search terms include "industrial gas cylinder nitrogen" or "industrial gas cylinder argon". Note that 

However, it is important to keep in mind that the gas cylinder detection model was specifically optimized for:
- industrial gas cylinders equipped with shrouds
- pictures with a viewpoint higher than the industrial gas cylinders.
- "non-LNG-shape" gas cylinders (nitrogen, oxygen, argon, hydrogen, etc).

Many online images of industrial gas cylinders may not depict shrouds (which is a very poor safety practice) and might have viewpoints lower than the top of the cylinder. The model may struggle a lot more with these images

## Content
We provide:
- the trained Mask R-CNN model
- Samples of synthetic dataset of various types
- A Demo Notebook to test the model with the real images provided (or with your own images)
