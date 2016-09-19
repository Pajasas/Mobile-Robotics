# Mobile-Robotics - house detection

Compiled opencv 3.1.0 for windows conda is available at http://www.ms.mff.cuni.cz/~tauferp/mr/ .
Put it into CONDA/Lib/site-packages/ directory.

# Task specification

# About used algorithm

My algorithm for house detection works in following steps:
- convert input to grayscale
- detect corners in input image
- group close corners together and filter candidates for house courners (will always be a few next to each other)
- construct graph where vertices correspond to candidates and edges are between candidates connected with dark line (fuzzy)
- find subgraph corresponding to a house
- draw 3d point base on its height and focal distance

TODO describe steps

# Demos

Common controls
- sliders
- windows

## Camera example

Controls

## Stored frames example

Controls
