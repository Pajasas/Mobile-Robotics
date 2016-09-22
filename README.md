# Mobile-Robotics - house detection

Compiled opencv 3.1.0 for windows conda is available at http://www.ms.mff.cuni.cz/~tauferp/mr/ .
Put it into CONDA/Lib/site-packages/ directory.

# Task specification

Locate an image of a house in a picture and draw a pyramid atop of it. House looks like this:

![Sample house](house.png?raw=true "Sample house")

# About used algorithm

My algorithm for house detection works in following steps:
- detect key house points in the image
- draw 3d point base on its height and focal distance

## House points detection

I've used standard conversion to from BGR to grayscale and applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to equalize contrast.
Opencv tutorial describing this technique can be found ![here](docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).

--todo picture of grayscaled image

Next step is to find key points of a house. 
My algorithm is looking for 6 key points marked in the following image.

![House key points](house_points.png?raw=true "House key points")

Here I've used corner detection cv2.cornerHarris to detect corners in the grayscale image.
From the structure of the house there should be about 3-5 corners next to each other for each point.
We can therefore discard both too small and too large groups of nearby corners.

--todo picture of corners image

Next 

--construct graph where vertices correspond to candidates and edges are between candidates connected with dark line (fuzzy)

--find subgraph corresponding to a house

# 3d drawing

We do not have calibration data for used camera.
Set focal distance by slider.

Opencv2 example: https://github.com/opencv/opencv/blob/master/samples/python/plane_ar.py .

# Demos

Common controls
- sliders
- windows

## Camera example

Controls

## Stored frames example

Controls
