# 3D Morphable Face Model

In this exercise, you will learn how to align a 3D Morphable Model (3DMM) to a 3D face scan using the Iterative Closest Point (ICP) and Gauss-Newton optimization algorithms.

The objective of this exercise is to minimize the difference between the model and the face scan, so that the morphable model accurately represents the face scan. The four graded tasks in this exercise are designed to test your understanding of different aspects of the alignment process. These tasks are:

1. __Shape matching__: Align the 3DMM to the face scan using procrustes analysis.
2. __Rigid ICP__: Perform rigid ICP alignment between the 3DMM and the face scan.
3. __Non-Rigid ICP__: Perform non-rigid ICP alignment between the morphable model and the face scan.
4. __(Bonus) Point-to-Plane optimization__: Implement the Point-to-Plane optimization technique to further improve the alignment results.

## Init
3D Morphable Model (3DMM) is a statistical model of the human face. It is a 3D shape model that can be used to represent a face in a compact form. The 3DMM is composed of a mean shape, a set of principal components (blend shapes).

In this homework we will use a very well known Basel Face Model (BFM). The BFM is a 3DMM that is trained on a large dataset of 3D face scans. The BFM is available for download from the [BFM website](faces.dmi.unibas.ch/bfm/bfm2017.html).

For your convenience, we have already downloaded the BFM model (you will download it when import the modules below). By downloading the BFM model, you agree to the terms of the [license](faces.dmi.unibas.ch/bfm/bfm2017.html). The BFM is represented as a BFM_layer class defined in BFM.py.

BFM model |  Scan
:-------------------------:|:-------------------------:
<img src="teaser_imgs/bfm_init.png" width="350"> |  <img src="teaser_imgs/scan_init.png" width="350"> 

**Problem**: The BFM model is not aligned with the face scan. We need to align (or "fit") the BFM model to the face scan using the Iterative Closest Point (ICP) and Gauss-Newton optimization algorithms. First we align it in a non-rigid way, then we will use the Point-to-Plane (or Point-to-Surface – bonus) optimization technique to further improve the alignment results.

<img src="teaser_imgs/init.png" width="550"> 

## Task 1: Rigid shape matching

__Task1__: Implement the `rigid_matching` function in `fitting_3DMM.py` to find the rigid transformation that aligns the scan to the BFM model. The function should return the transformation matrix.

<img src="teaser_imgs/procrustus.png" width="550"> 

## Task 2: Rigid ICP

__Task 2__: Implement the function `rigid_ICP` in the `fitting_3DMM.py` file.

<!-- <img src="teaser_imgs/icp_rigid.png" width="550">  -->
<img src="teaser_imgs/rigid_icp.gif" width="550"> 

## Task 3: Non-Rigid Fitting

Scan & BFM Overlay | BFM
:-------------------------:|:-------------------------:|
<img src="teaser_imgs/non_rigid_overlay_point2point.gif" width="350"> |  <img src="teaser_imgs/non_rigid_bmf_point2point.gif" width="350"> |   

## Task 4: Point-to-Plane ICP

Scan & BFM Overlay | BFM
:-------------------------:|:-------------------------:|
<img src="teaser_imgs/non_rigid_overlay_point2plain.gif" width="350"> |  <img src="teaser_imgs/non_rigid_bmf_point2plain.gif" width="350"> |   
