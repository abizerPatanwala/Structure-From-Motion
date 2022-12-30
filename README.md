# Structure-from-motion
This project implements **Strucutre from Motion(SFM)** for 5 input images in the *P3Data* folder. The input images are already distortion corrected. *Caliberation*
folder contains *cameraParams.mat* file which has camera parameters. *output* folder contains output images from various stages of structure from motion.
*points.png* file visualizes the projection of world points of the scene observed from top view.

**SFM** is used for 3d recontruction of data from 2d images. It is assumed that the first camera pose is origin. Since the scale of the scene 
cannot be recovered, the scene is constructed only upto a scale. SFM mainly involes following stages:
1. Initialisation: This stage involves initialising the world points and the camera pose from first two images. First, features of both the images are extracted
and matched. The features are usually corners in an image and the descriptors used to describe these features can be SIFT,SURF, ORB or any other descriptors.
Then, **RANSAC** is used to filter the outliers and only get the correct matches. Then, **Fundamental Matrix** is calculated using the correct matches. *DLT* 
is used to compute Fundamental Matrix. For more accuracy, non linear optimization methods can also be used for computing fundamental matrix. 
But even for that an initial estimate is required, which comes from DLT. In this project, Fundamental Matrix from DLT is directly used without any further optimization.
**Essential Matrix** is extracted from Fundamental Matrix and we get 4 possible camera poses from esential Matrix. To get the correct pose, we **triangulate**
the points using each possible camera pose and using **chirality condition** determine the correct pose. The correct pose is the one, which has maximum points in fron of it.
