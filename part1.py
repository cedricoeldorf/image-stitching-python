import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import linear_model
from utilities import *


def ImStitch(filename_1, filename_2,type, error_threshold, distance_thresh, sample_size):
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)

    ######################
    ## Extract Keypoints and Descriptors
    d1, d2, coords1, coords2 = get_sift(img_1,img_2)
    plot_matches(img_1,img_2,coords1,coords2, 'pre_threshold')

    #########################################################
    # Get Best Matches
    ## DO SENSITIVITY ANALYSIS ON DISTANCE TYPE
    index_1, index_2, distances = get_top_matches(d1,d2, distance_thresh,type)
    while len(index_1) > 5000:
        print("##############################################")
        print("REDUCING DISTANCE THRESHOLD TO ADJUST TO IMAGE")
        distance_thresh = distance_thresh*0.8
        print(distance_thresh)
        index_1, index_2, distances = get_top_matches(d1,d2, distance_thresh,type)

    ## Extract best matches
    coords1 = np.array(coords1)
    matches_1 = coords1[index_1]
    coords2 = np.array(coords2)
    matches_2 = coords2[index_2]

    plot_matches(img_1,img_2,matches_1,matches_2, 'pre_ransac')


    ############################
    ## RANSAC
    ############################
    # Number of iterations

    p = 0.99
    e = 0.05

    N = np.log(1 - p) / np.log(1-(1-e)**sample_size)
    N = int(N) + 5

    ###########################
    ## RANSAC ITERATIONS
    homogrogaphy_matrices, removal_matches, all_inliers, n_inliers, inlier_residual = run_RANSAC(N, matches_1, matches_2, sample_size, error_threshold)
    while np.max(n_inliers) > 800:
        print("##############################################")
        print("REDUCING ERROR THRSHOLD")
        error_threshold = error_threshold*0.85
        print(error_threshold)
        homogrogaphy_matrices, removal_matches, all_inliers, n_inliers, inlier_residual = run_RANSAC(N, matches_1, matches_2, sample_size, error_threshold)




    ######################
    ## Remove bad keypoints based on RANSAC RESULTS
    ######################
    outliers = removal_matches[np.argmax(n_inliers)].ravel()
    opt_matches_1 = np.asarray([i for j, i in enumerate(matches_1) if j not in outliers])
    opt_matches_2 = np.asarray([i for j, i in enumerate(matches_2) if j not in outliers])
    opt_H = homogrogaphy_matrices[np.argmax(n_inliers)]


    #########################################
    ## Solve for Homography Matrix using big matrices
    H = final_homography(opt_matches_1,opt_matches_2)


    ## Show Inliers on both images
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)
    plot_matches(img_1,img_2,opt_matches_1,opt_matches_2, 'inliers')

    ## STITCH IMAGE
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)
    res = get_stitched_image(img_2,img_1, H)

    ## GET ERROR VALUE
    X = np.matrix([opt_matches_1.T[0], opt_matches_2.T[1], np.ones(len(opt_matches_2))])
    Y_pred = np.dot(H,X)
    Y_pred = Y_pred[0:2].T
    Y_pred = Y_pred / np.linalg.norm(Y_pred)
    opt_matches_2 = opt_matches_2  / np.linalg.norm(opt_matches_2)
    distances = cdist(Y_pred, opt_matches_2,'euclidean')
    errors = distances.diagonal()

    print("__________________________________")
    print("Please see final image saved in working directory")
    #
    # img3 = cv2.drawMatches(img_1,coords1,img_2,coords2, np.array((opt_matches_1, opt_matches_2))
    # ,flags=2,outImg = img_1)
    return res
