import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import linear_model


## MAIN ransac function. Not iterative.
def RANSAC(matches_1, matches_2, s, error_threshold):

    ## Take random sample of matches
    sample_ind = np.random.randint(len(matches_1), size=(s,1))

    match_1_sample = matches_1[sample_ind]
    match_2_sample = matches_2[sample_ind]
    match_1_sample = np.asarray(match_1_sample)
    match_2_sample = np.asarray(match_2_sample)

    #########################################
    ## Solve for Homography Matrix using big matrices
    X = np.matrix([match_1_sample.T[0][0], match_1_sample.T[1][0], np.ones(s)])
    Y = np.matrix([match_2_sample.T[0][0], match_2_sample.T[1][0],np.ones(s)])

    ############################################
    #########################################
    ## CONSTRUCT MATRICES
    sum_x_sq = np.square(X[0]).sum()
    sum_x_y = np.multiply(X[0],X[1]).sum()
    sum_x = X[0].sum()
    sum_y_sq = np.square(X[1]).sum()
    sum_y = X[1].sum()

    a = np.array([
    [sum_x_sq, sum_x_y, sum_x, 0, 0, 0],
    [sum_x_y, sum_y_sq, sum_y, 0, 0, 0],
    [sum_x, sum_y, s, 0, 0, 0],
    [0, 0, 0,sum_x_sq, sum_x_y, sum_x],
    [0, 0, 0,sum_x_y, sum_y_sq, sum_y],
    [0, 0, 0,sum_x, sum_y, s]
    ])

    sum_u_x = np.multiply(Y[0],X[0]).sum()
    sum_u_y = np.multiply(Y[0],X[1]).sum()
    sum_u = Y[0].sum()
    sum_v_x = np.multiply(Y[1],X[0]).sum()
    sum_v_y = np.multiply(Y[1],X[1]).sum()
    sum_v = Y[1].sum()

    b = np.array([sum_u_x, sum_u_y,sum_u,sum_v_x,sum_v_y,sum_v])
    ############################################
    ############################################

    ## Calculate H matrix
    a_matrix = np.linalg.solve(a,b)
    H = np.array([
    [a_matrix[0],a_matrix[1],a_matrix[2]],
    [a_matrix[3],a_matrix[4],a_matrix[5]],
    [0,0,1]])

    ### take the new H matrix and predict image 2 points using img 1 points
    X = np.matrix([matches_1.T[0], matches_1.T[1], np.ones(len(matches_1))])
    Y_pred = np.dot(H,X)

    ############################
    # the squared distance
    # between the point coordinates in one image, and the transformed coordinates of
    # the matching points in the other image

    # filter inliers and outliers, return outlier indices and statistics
    ############################

    Y_pred = Y_pred[0:2].T
    Y_pred = Y_pred / np.linalg.norm(Y_pred)
    matches_2 = matches_2  / np.linalg.norm(matches_2)
    distances = cdist(Y_pred, matches_2,'euclidean')
    errors = distances.diagonal()
    inliers = np.where(errors<error_threshold)[0]
    outliers = np.where(errors>=error_threshold)[0]
    n_inliers = inliers.shape[0]
    n_outliers = outliers.shape[0]
    outlier_index = np.argwhere(errors>=error_threshold)
    inlier_residual = inliers.mean()

    # print("###")
    # print("Number of Inliers")
    # print(n_inliers)

    return H, outlier_index, inliers, outliers, inlier_residual


## Visually compare image 1 and 2
def plot_matches(img_1,img_2,matches_1,matches_2, name):

    for match in matches_1:
        cv2.circle(img_1,(int(match[0]),int(match[1])),10,(0,0,255),1)
    for match in matches_2:
        cv2.circle(img_2,(int(match[0]),int(match[1])),10,(0,0,255),1)
    cv2.imwrite('./match1_' + str(name) + '.png',img_1)
    cv2.imwrite('./match2_' + str(name) + '.png',img_2)


## Get descriptors and coordinates
def get_sift(img_1,img_2):
    # sift = cv2.xfeatures2d.SURF_create(400)
    # sift.setExtended(True)
    sift = cv2.xfeatures2d.SIFT_create()
    k1, d1 = sift.detectAndCompute(img_1, None)
    k2, d2 = sift.detectAndCompute(img_2, None)

    ## Normalize for 0 mean and unit std
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    ## Extract coordinates of keypoints
    coords1 = []
    coords2 = []
    for i in range(len(k1)):
        coords1.append(k1[i].pt)
    for i in range(len(k2)):
        coords2.append(k2[i].pt)
    return d1, d2, coords1, coords2

## Get closest matches based on distacne
def get_top_matches(desc_1, desc_2, threshold, type):

    if type == 'euc':
        desc_1 = desc_1 / np.linalg.norm(desc_1)
        desc_2 = desc_2  / np.linalg.norm(desc_2)
        distances = cdist(desc_1, desc_2)
    if type == 'corr':
        distances = np.zeros((len(desc_1), len(desc_2)))
        for i in range(len(desc_1)):
            for j in range(len(desc_2)):
                distances[i][j] = np.correlate(desc_1[i], desc_2[j])
        distances = distances / distances.max()
        distances = 1 - distances

    index_1 = []
    index_2 = []
    for i in range(0,len(distances)):
        if distances[i].min() < threshold:
            index_1.append(i)
            index_2.append(distances[i].argmin())

    return index_1, index_2, distances

## Run RANSAC iterations
def run_RANSAC(N, matches_1, matches_2, sample_size, error_threshold):
    removal_matches = []
    all_inliers = []
    homogrogaphy_matrices = []
    n_inliers = []
    for n in range(N):
        H, outlier_index, inliers, outliers, inlier_residual = RANSAC(matches_1, matches_2, sample_size, error_threshold)
        homogrogaphy_matrices.append(H)
        removal_matches.append(outlier_index)
        all_inliers.append(inliers)
        n_inliers.append(len(inliers))
    print("Optimal Inlier count:")
    print(np.max(n_inliers))
    print("Average Inlier Residual:")
    inlier_residual = np.mean(all_inliers[np.argmax(n_inliers)])
    print(inlier_residual)
    print("Number of Outliers:")
    n_outlier = len(matches_1) - np.max(n_inliers)
    print(n_outlier)

    return homogrogaphy_matrices, removal_matches, all_inliers, n_inliers, inlier_residual

## Calculate the final H matrix
def final_homography(opt_matches_1,opt_matches_2):
    X = np.matrix([opt_matches_1.T[0], opt_matches_1.T[1], np.ones(len(opt_matches_1))])
    Y = np.matrix([opt_matches_2.T[0], opt_matches_2.T[1],np.ones(len(opt_matches_1))])

    ############################################
    #########################################
    ## CONSTRUCT MATRICES
    sum_x_sq = np.square(X[0]).sum()
    sum_x_y = np.multiply(X[0],X[1]).sum()
    sum_x = X[0].sum()
    sum_y_sq = np.square(X[1]).sum()
    sum_y = X[1].sum()

    a = np.array([
    [sum_x_sq, sum_x_y, sum_x, 0, 0, 0],
    [sum_x_y, sum_y_sq, sum_y, 0, 0, 0],
    [sum_x, sum_y, len(opt_matches_1), 0, 0, 0],
    [0, 0, 0,sum_x_sq, sum_x_y, sum_x],
    [0, 0, 0,sum_x_y, sum_y_sq, sum_y],
    [0, 0, 0,sum_x, sum_y, len(opt_matches_1)]
    ])

    sum_u_x = np.multiply(Y[0],X[0]).sum()
    sum_u_y = np.multiply(Y[0],X[1]).sum()
    sum_u = Y[0].sum()
    sum_v_x = np.multiply(Y[1],X[0]).sum()
    sum_v_y = np.multiply(Y[1],X[1]).sum()
    sum_v = Y[1].sum()

    b = np.array([sum_u_x, sum_u_y,sum_u,sum_v_x,sum_v_y,sum_v])
    ############################################
    ############################################

    ## Calculate H matrix
    a_matrix = np.linalg.solve(a,b)
    H = np.array([
    [a_matrix[0],a_matrix[1],a_matrix[2]],
    [a_matrix[3],a_matrix[4],a_matrix[5]],
    [0,0,1]])

    return H

######################################
## NOTE THIS FUNCTION IS CURTEST OF
## https://github.com/pavanpn/Image-Stitching/blob/master/stitch_images.py
## Taken as alternative to matlabs warp and transform functions
def get_stitched_image(img1, img2, M):

    # Get width and height of input images
    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min,-y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
    							[0, 1, transform_dist[1]],
    							[0,0,1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),(x_max-x_min, y_max-y_min))
    result_img[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img1
    cv2.imwrite('./result.png', result_img)
    # Return the result
    return result_img
