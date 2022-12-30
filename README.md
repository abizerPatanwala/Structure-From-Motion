# Structure-From-Motion
## Overview ##
[^1]: The input images, camera parameters and matching.txt files in the *P3Data* folder was provided by [Professor Nitin Sanket](https://github.com/nitinjsanket/) as part of RBE 549:Computer Vision graduate course from RBE department of Worcester Polytechnic Institute(WPI) 
This project implements **Strucutre from Motion(SFM)** for 5 input images in the *P3Data* folder. The input images are already distortion corrected. *Caliberation*
folder contains *cameraParams.mat* file which has camera parameters. *output* folder contains output images from various stages of structure from motion.
*points.png* file visualizes the projection of world points of the scene observed from top view.

**SFM** is used for 3d recontruction of data from 2d images. It is assumed that the first camera pose is origin. Since the scale of the scene 
cannot be recovered, the scene is constructed only upto a scale. SFM mainly involes following stages:
1. **Initialisation**: This stage involves initialising the world points and the camera pose from first two images. First, features of both the images are extracted
and matched. The features are usually corners in an image and the descriptors used to describe these features can be SIFT,SURF, ORB or any other descriptors.
Then, **RANSAC** is used to filter the outliers and only get the correct matches. Then, **Fundamental Matrix** is calculated using the correct matches. *DLT* 
is used to compute Fundamental Matrix. For more accuracy, non linear optimization methods can also be used for computing fundamental matrix. 
But even for that an initial estimate is required, which comes from DLT. In this project, Fundamental Matrix from DLT is directly used without any further optimization.
**Essential Matrix** is extracted from Fundamental Matrix and we get 4 possible camera poses from esential Matrix. To get the correct pose, we **triangulate**
the points using each possible camera pose and using **chirality condition** determine the correct pose. The correct pose is the one, which has maximum points in fron of it.

2. **Perspective N Point(PnP)**: Once we have, initial poses and map, for calculating further poses we use PnP algorithm. For each new image, we match it with previous *n* images and find the world points associated with current image. Then using PnP we calculate the pose of the current image using DLT. Using this initial estimate of the pose, the pose is further refined using **Levenberg–Marquard(LM)** algorithm.

3. **Triangulation**: Once we have the pose of the current image, we triangulate matched features whose world points are not known. Initial estimate from the DLT algorithm is used to further refine the world points using LM algorithm.

4. **Bundle Adjustment**: Finally the camera pose and the world points related to the current image are jointly optimised using LM algorithm. Then for each new image steps 2-4 is repeated.

## Setup
Download all .py files in the same folder. Download the images from *P3Data* in a folder with same name, in the directory of .py files. On Ubuntu terminal run `python3 main.py` which runs SFM and generates a file named *world_pts.txt*, which contains the world points from SFM. Then run `plot.py` to visualise the  projection of world points of the scene observed from top view.   
