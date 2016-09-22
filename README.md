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

I've used standard conversion to from BGR to gray-scale and applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to equalize contrast.
Opencv tutorial describing this technique can be found ![here](docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).

!["Gray-scaled image"](clahe.png?raw=true "Gray-scaled image")

Next step is to find key points of a house. 
My algorithm is looking for 6 key points marked in the following image.

![House key points](house_points.png?raw=true "House key points")

Here I've used corner detection cv2.cornerHarris to detect corners in the grayscale image.
From the structure of the house there should be about 3-5 corners next to each other for each point.
We can therefore discard both too small and too large groups of nearby corners.
This steps creates a set of candidates for house key points.

![corners](corners.png?raw=true "Corners, key points are marked in red circle")

Next task is to construct a graph where vertexes are candidates from previous steps.
Edge is added between each two candidates that are connected with dark enough line.

!["Graph"](graph.png?raw=true "Graph")

Final step here is to find a subgraph corresponding to a house.

?Describe the subgraph search in detail?

If there is a subgraph equal to a house graph the subgraph points a passed to be drawn in 3d.

# 3d drawing

We often do not have calibration data for the used camera, so this algorithm uses variable focal distance set by a slider.
Calibration data could be used too to increase accuracy.

Because we know relative distances between the house key points we can use cv2.solvePnP to create a projection from 3d world space to our 2d plane.
Then we use this projection to get 2d coordinates for the middle point of the pyramid in specified height.

!["Final point of a pyramid placed"](3d.png?raw=true "Final point of a pyramid placed")

Opencv2 example: https://github.com/opencv/opencv/blob/master/samples/python/plane_ar.py .

# Demos

Code is provided as a jupyter notebook for python 2.7 in file Pyramid.ipynb .

There are also two examples as python files:
- example_camera.py - run house detection on camera feed.
- example_frames.py - run house detection on a set of frames used for testing.

Following windows show up after starting:
- clahe - grayscaled version of the current frame
- markers_col - detected corners with marked key point candidates
- graph - constructed graph
- dst - main window with a house draw (if detected)

Main window also provides following sliders:
- focal - focal distance
- height - height of the top of the pyramid
- max_mean - how dar the lines need to be to be considered as edges in the graph creation

## Camera example controls
- q - stop example

## Stored frames example controls
- q - stop example
- space - next frame
