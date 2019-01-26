import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from utilities import *

## NOTE THIS SCRIPT IS PURELY FOR SENSITIVITY ANALYSIS
type = 'euc'
distance_thresh = 0.01
error_threshold = 0.001
sample_size = 10
def visualize(x,y,title,xlab,ylab):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig('./' + title + '.png', bbox_inches='tight')
    plt.close()

def run_all(type, distance_thresh, error_threshold, sample_size):
    filename_1 = './b.jpg'
    filename_2 = './a.jpg'
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)

    ######################
    ## Extract Keypoints and Descriptors
    d1, d2, coords1, coords2 = get_sift(img_1,img_2)

    #########################################################
    # Get Best Matches
    ## DO SENSITIVITY ANALYSIS ON DISTANCE TYPE
    print(distance_thresh)
    print(type)
    index_1, index_2, distances = get_top_matches(d1,d2, distance_thresh,type)



    ## Extract best matches
    coords1 = np.array(coords1)
    matches_1 = coords1[index_1]
    coords2 = np.array(coords2)
    matches_2 = coords2[index_2]


    ############################
    ## RANSAC
    ############################
    # Number of iterations

    p = 0.99
    e = 0.05

    N = np.log(1 - p) / np.log(1-(1-e)**sample_size)
    N = int(N) + 10

    ###########################
    ## RANSAC ITERATIONS
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


    #H, mask = cv2.findHomography(opt_matches_1, opt_matches_2)


    X = np.matrix([opt_matches_1.T[0], opt_matches_2.T[1], np.ones(len(opt_matches_2))])
    Y_pred = np.dot(H,X)
    Y_pred = Y_pred[0:2].T
    Y_pred = Y_pred / np.linalg.norm(Y_pred)
    opt_matches_2 = opt_matches_2  / np.linalg.norm(opt_matches_2)
    distances = cdist(Y_pred, opt_matches_2,'euclidean')
    errors = distances.diagonal()
    errors = errors.mean()
    return errors


select = ['types', 'distance_thresh', 'error_threshold', 'sample_size']
for select_analysis in select:
    if select_analysis == 'types':
        types = ['euc', 'corr']
        plot_y = []
        for loop in types:
            s = []
            for i in range(5):
                if loop == 'corr':
                    distance_thresh = 0.9
                error = run_all(loop, distance_thresh, error_threshold, sample_size)
                s.append(error)
            plot_y.append(np.mean(s))
        plt.bar(types, plot_y)
        plt.title('Sensitivity Analysis: Matching Distance')
        plt.xlabel('Metric')
        plt.ylabel('Error')
        plt.savefig('./' + 'Sensitivity Analysis: Matching Distance' + '.png', bbox_inches='tight')
        plt.close()
    if select_analysis == 'distance_thresh':
        distance_thresh = np.arange(0.05, 0.5, 0.05)
        plot_y = []
        for loop in distance_thresh:
            s = []
            for i in range(5):
                error = run_all(type, loop, error_threshold, sample_size)
                s.append(error)
            plot_y.append(np.mean(s))
        distance_thresh = distance_thresh.tolist()
        visualize(distance_thresh,plot_y,'Sensitivity Analysis: Distance Threshold','Threshold','Error')
    if select_analysis == 'error_threshold':
        distance_thresh = 0.01
        error_threshold = 0.001
        sample_size = 10
        error_threshold = np.arange(0.001, 0.02, 0.001)
        plot_y = []
        for loop in error_threshold:
            s = []
            for i in range(5):
                error = run_all(type, distance_thresh, loop, sample_size)
                s.append(error)
            plot_y.append(np.mean(s))
        error_threshold = error_threshold.tolist()
        visualize(error_threshold,plot_y,'Sensitivity Analysis: RANSAC Error Threshold','Threshold','Error')
    if select_analysis == 'sample_size':
        distance_thresh = 0.01
        error_threshold = 0.001
        sample_size = 10
        sample_size = range(4,15,1)
        plot_y = []
        for loop in sample_size:
            s = []
            for i in range(10):
                error = run_all(type, distance_thresh, error_threshold, loop)
                s.append(error)
            plot_y.append(np.mean(s))

        sample_size = list(sample_size)
        visualize(sample_size,plot_y,'Sensitivity Analysis: RANSAC Sample Size','Sample Size','Error')


#######################
## READ IMAGES
