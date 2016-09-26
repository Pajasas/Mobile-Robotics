# Mobile-Robotics - house detection

Compiled opencv 3.1.0 for windows conda is available at http://www.ms.mff.cuni.cz/~tauferp/mr/ .
Put it into CONDA/Lib/site-packages/ directory.

# Task specification

Locate an image of a "house" in a picture and draw a pyramid atop of it. House looks like this:

![Sample house](house.png?raw=true "Sample house")

# About used algorithm

My algorithm for house detection works in following steps:
- detect key house points in the image
- draw 3d point base on its height and focal distance

## House points detection

I've used standard conversion from BGR to gray-scale and applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to equalize contrast.
Opencv tutorial describing this technique can be found ![here](docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).

!["Gray-scaled image"](clahe.png?raw=true "Gray-scaled image")

Next step is to find key points of the house. 
My algorithm is looking for 6 key points marked in the following image.

![House key points](house_points.png?raw=true "House key points")

Here I've used corner detection cv2.cornerHarris to detect corners in the gray-scale image.
From the structure of the house there should be about 3-5 corners next to each other for each key point.
We can therefore discard both too small and too large groups of nearby corners.
I've tried two ways of accomplishing this:

One way would be to group corners by distance between each other.
However this was too slow to be ran on a live feed when implemented in Python.

Next way is to use a temporary B/W image and draw small white circles for each corner.
Then OpenCV2 function cv2.connectedComponentsWithStats can be used to locate continuous regions in this image and their properties like size and centrepoint.
Benefit here is that the heavy calculation is done inside the library and not in interpreted Python.

The candidates for key points are the centrepoints of connected components with appropriate size.
Exact size boundaries and white circles radius depend on the image resolution and "house" proportions (line width, conrner size).

In the following picture are circles around each detected corner with same color per each connected component.
Red circle marks components that are key point candidates.
The values used in the following image were:
- circle radius - 2px
- minimum width/height - 4*radius
- maximum width/height - 11*radius

![corners](corners.png?raw=true "Corners, key points are marked in red circle")

Next task is to construct a graph where vertexes are candidates from previous steps.
Edge is added between each two candidates that are connected with dark enough line.

!["Graph"](graph.png?raw=true "Graph")

Final step here is to find a subgraph corresponding to a house.
Let's take a look again at how the house should look and what graph it should have.

![House key points](house_points.png?raw=true "House key points")

The top node (orange) has a degree of 2 and is connected to 2 other nodes (red).
Those are then both connected to each other and to 3 other nodes (blue and green) so they have a degree of 5.
Green and both blue nodes are also connected to each other.

I've used following steps to check for this type of subgraph.
Following logic is done for each node n in the graph:
- if n does not have a degree of 2 => n is not a top of a house
- a, b = neigbours of n 
- if a and b don't have a degree of 5 => n is not a top of a house
- check orientation of the top triangle (a,b,n) and swap a and b if needed
- x = neighbour of a that has shortes sum of edges #this will be the green point if this is a house subgraph
- c,d = neighbours of a,b that are not a,b,n or x. if there are more than 2 neighbours => n is not top of a house
- check orientation of the bottom triangle (x,c,d) and swap c and d if needed
- house subgraph found, draw top point in 3d for house with key points (n,a,b,x,c,d)

# 3d drawing

I didn't have calibration data for the used camera, so I've used variable focal distance set by a slider.
Calibration data could be used as well to increase accuracy.

Because we know relative distances between key points of the house we can use cv2.solvePnP to create a projection from 3d world space to our 2d plane.
Then we use this projection to get 2d coordinates for the middle point of the pyramid in specified height.

!["Final point of a pyramid placed"](3d.png?raw=true "Final point of a pyramid placed")

Opencv2 example: https://github.com/opencv/opencv/blob/master/samples/python/plane_ar.py .

# Demos

Code is provided as a jupyter notebook for python 2.7 in file Pyramid.ipynb .

There are also two examples as python files:
- example_camera.py - run the house detection algorithm on a camera feed.
- example_frames.py - run the house detection algorithm on a set of frames used for testing.

Following windows will show up after running the examples:
- clahe - gray-scaled version of the current frame
- markers_col - detected corners with marked key point candidates
- graph - constructed graph
- dst - main window with a house draw (if detected)

Main window also provides following sliders:
- focal - focal distance
- height - height of the top of the pyramid
- max_mean - how dark the lines need to be to be considered as edges in the graph creation

## Camera example controls
- q - stop example

## Stored frames example controls
- q - stop example
- space - next frame
