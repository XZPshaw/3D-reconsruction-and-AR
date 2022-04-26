import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def find_homography(X, X_P, NrPoints,norm='l1', normalization=True):
    
    """if normalization procedure is required"""
    if normalization:
        """compute the center the corresponding points on both images"""
        X_center = np.mean(X, axis=1).reshape(3, 1)
        X_P_center = np.mean(X_P, axis=1).reshape(3, 1)
    
        #print("X_center:", X_center)
        #print("X_P_center:", X_P_center)        
        
        """compute the translation matrix to perform decentralized step in order to form the composite T normal"""
        X_center = np.mean(X, axis=1).reshape(3, 1)
        X_P_center = np.mean(X_P, axis=1).reshape(3, 1)        
        X_trans = np.array([[1, 0, 0 - X_center[0]], [0, 1, 0 - X_center[1]], [0, 0, 1]], dtype=float)
        X_P_trans = np.array([[1, 0, 0 - X_P_center[0]], [0, 1, 0 - X_P_center[1]], [0, 0, 1]], dtype=float)

        #print("X_trans:", X_trans)
        #print("X_P_trans:", X_P_trans)
        
        """compute decentralized points in both image"""

        X_decentered = X - X_center
        X_P_decentered = X_P - X_P_center

        #print("X_decentered:", X_decentered)
        #print("X_P_decentered:", X_P_decentered)
        
        """compute sum of distance between points in each image from their center, the scale them by sqrt(2)/current sum distance in order to keep the distance as sqrt(2)"""

        X_sum_dis = np.sum(np.abs(X_decentered))
        X_P_sum_dis = np.sum(np.abs(X_P_decentered))

        #print("X_sum_dis:", X_sum_dis)
        #print("X_P_sum_dis:", X_P_sum_dis)

        X_scale_factor = np.sqrt(2) / X_sum_dis
        X_P_scale_factor = np.sqrt(2) / X_P_sum_dis

        #print("X_scale_factor:", X_scale_factor)
        #print("X_P_scale_factor:", X_P_scale_factor)

        X_scale = np.array([[X_scale_factor, 0, 0], [0, X_scale_factor, 0], [0, 0, 1]])
        X_P_scale = np.array([[X_P_scale_factor, 0, 0], [0, X_P_scale_factor, 0], [0, 0, 1]])

        #print("X_scale:", X_scale)
        #print("X_P_scale:", X_P_scale)
        
        """formulate the normalization matrix in both images"""
        T_norm = X_scale.dot(X_trans).astype(float)
        T_P_norm = X_P_scale.dot(X_P_trans)

        # print("T_norm:", T_norm)
        # print("T_P_norm:", T_P_norm)
        
        """
        apply the normalization transformation to the choosen pairs of points
        """
        X = T_norm.dot(X)
        X_P = T_P_norm.dot(X_P)

        # print("normalized_X:", X)
        # print("normalized_X_P:", X_P)

    """
    therefore, if there is normalization flag,the X and X_P will be the coordinates normalized points and it not, they will be the original coordinates during selection 
    """
    
    
    A = np.zeros((NrPoints * 2, 9))
    
    """
    if the norm flag is l1
    we will keep iteration on update n and a until they converges, here I simply keep iteration 100 times. By exeriment, they would converge in less than 100 iterations
    
    """
    if norm == 'l1':
        n = np.ones((NrPoints * 2))
        diag = np.identity(NrPoints * 2)
        print("diag", diag)

        for t in range(100):
            for i in range(NrPoints):

                x = X[:, i]
                #print("x", x)
                x_prime = X_P[:, i]
                
                A1 = np.zeros((2, 9))
                A1[0, 3:6] = -1 * x_prime[2] * x
                A1[0, 6:] = x_prime[1] * x
                A1[1, 0:3] = x_prime[2] * x
                A1[1, 6:] = -1 * x_prime[0] * x

                # print("A1:",A1)
                A[i * 2:(i + 1) * 2, :] = A1
                # print("A:",A)
            ### derive h
            A = diag.dot(A)
            result = np.linalg.svd(A)

            # print(result)
            v = result[2].T
            h = v[:, -1]
            H = h.reshape(3, 3)

            # update n = a_i * h
            n = np.absolute(A.dot(h))
            print("n:", n)
            n = np.reciprocal(np.sqrt(n))
            np.fill_diagonal(diag, n)
            print("updated diag:", n)
            """if the norm flag, we will use euclidean norm, just formulate A matrix by construct each pairs of NrPoints"""
    elif norm == 'euclidean':
        for i in range(NrPoints):
            x = np.ones((3,))
            x_prime = np.ones((3,))

            # x[:2] = ref1[i]
            x = X[:, i]
            #print("x", x)
            # x_prime[:2] = ref2[i]
            x_prime = X_P[:, i]
            #print("x prime", x_prime)

            A1 = np.zeros((2, 9))
            A1[0, 3:6] = -1 * x_prime[2] * x
            A1[0, 6:] = x_prime[1] * x
            A1[1, 0:3] = x_prime[2] * x
            A1[1, 6:] = -1 * x_prime[0] * x

            #print("A1:", A1)
            A[i * 2:(i + 1) * 2, :] = A1
    else:
        print("invalid norm form")
        return

    #print("A:", A)
    
    """compute SVD of A and the h matrix will be the last column of V from SVD decomposition"""
    result = np.linalg.svd(A)
    v = result[2].T
    #print("v:", v)

    h = v[:, -1]
    
    """reshape h to 3 by 3 to formulate the final homography"""
    H = h.reshape(3, 3)
    
    
    """
    if the normalization step is performed before, it is expected to perform the denomalize step to find the final homography for images in original scale`
    """
    if normalization:
        H = (np.linalg.inv(T_P_norm)).dot(H).dot(T_norm)
        #denomalized_H = (np.linalg.inv(T_P_norm)).dot(H).dot(T_norm)
    #print("Homography:", denomalized_H)
    
    return H

"""
draw the warped image after finding homography
"""
def apply_homography(img1, img2, H, fit_origin=False, get_image=False):
    height = img1.shape[0]
    width = img1.shape[1]
    P = np.array([[0, width, width, 0], [0, 0, height, height], [1, 1, 1, 1]])

    P_prime_homo = H.dot(P)
    #print("P prime homo", P_prime_homo)

    P_prime = np.array([[P_prime_homo[0, :] / P_prime_homo[2, :]], [P_prime_homo[1, :] / P_prime_homo[2, :]]])
    #print("P prime", P_prime)

    x_min = np.min(P_prime[0, :])
    y_min = np.min(P_prime[1, :])
    x_max = np.max(P_prime[0, :])
    y_max = np.max(P_prime[1, :])

    offset = np.array([[1, 0, -1 * x_min], [0, 1, -1 * y_min], [0, 0, 1]])
    # Apply the perspective transformation to the image
    
    out = cv2.warpPerspective(img1, offset.dot(H), (int(x_max-x_min), int(y_max-y_min)), flags=cv2.INTER_LINEAR)
    
    if fit_origin == True:
        #print("img2:",img2.shape)
        out = cv2.warpPerspective(img1, H, (img2.shape[1],img2.shape[0]), flags=cv2.INTER_LINEAR)
        #out = cv2.warpPerspective(img1, H, (1500,1500), flags=cv2.INTER_LINEAR)        
    if get_image == True:
        return out
    
    #print(out.shape)
    """Display the transformed image"""
    plt.imshow(out, cmap='gray')
    plt.show()
    


if __name__ == '__main__':
    NrPoints = 5

    path1 = "./key/key1.jpg"
    img1 = cv2.imread(str(path1))
    plt.imshow(img1)
    ref1 = plt.ginput(NrPoints)

    print(ref1)
    path2 = "./key/key3.jpg"
    img2 = cv2.imread(str(path2))
    plt.imshow(img2)
    ref2 = plt.ginput(NrPoints)
    print(ref2)

    X = np.ones((3, NrPoints))
    X_P = np.ones((3, NrPoints))
    
    for i in range(NrPoints):
        X[:2,i] = ref1[i]
        X_P[:2,i] = ref2[i]

    H1 = find_homography(X, X_P, NrPoints, norm='euclidean', normalization = False) 
    H2 = find_homography(X, X_P, NrPoints, norm='l1', normalization=False)
    denomalized_H1 = find_homography(X, X_P, NrPoints, norm='euclidean', normalization = True)
    denomalized_H2 = find_homography(X, X_P, NrPoints, norm='l1', normalization=True)
    
    
    
    apply_homography(img1, img2, H1)
    apply_homography(img1, img2, H2)  
    apply_homography(img1, img2, denomalized_H1)    
    apply_homography(img1, img2, denomalized_H2)
