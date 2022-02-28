
# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum, Anastasija Karpova, Malte Pedersen, Stefan Hein Bengtson, and Thomas B. Moeslund

Implementation of popular metric learning based  person re-identification methods.
If the method is a reimplementation, it will be clearly stated
"""

import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA


def _calc_PCA(X, dim=None, ratio=None):
    """
    Calculates the Principal Componenet Analysis transformation for the input data

    Input:
        X: Input matrix of size [n_samples, n_features] which is used to determine the PCA transformation
        dim: The number of dimensions the data is projected down onto. Default is None, meaning min(n_samples-1, n_features) dimensions are found
        ratio: The ratio of the explained variance which should be used. Default is None, meaning 100% is used. Should be supplied as a floating point number between 0. and 1.

    Output:
        pca: An object holding the PCA transformation
    """

    if dim is not None:
        assert X.shape[1] >= dim, ("The dimensions argument should be smaller than the number of features in X.\nYou supplied {}, while there are {} features in X".format(
            dim, X.shape[1]))
    if ratio is not None:
        assert ratio <= 1.0 and ratio >= 0.0, (
            "The given ratio is not in the range [0,1], but {}".format(ratio))

    pca = PCA(n_components=dim)

    try:
        pca.fit(X)
    except np.linalg.LinAlgError as err:
        return False

    if ratio is not None:
        if ratio < 1.0:
            explained_variance = pca.explained_variance_ratio_.cumsum()
            over_threshold = explained_variance >= ratio
            n_dim = np.where(over_threshold)[0][0] + 1
        else:
            explained_variance = pca.explained_variance_ratio_.cumsum()
            over_threshold = np.isclose(explained_variance, 1.0)
            # take the lowest index where 100% variance is explained, use the one before it
            n_dim = np.where(over_threshold)[0][0]
        pca.components_ = pca.components_[:n_dim]
        print("{} % variance was found with {} components".format(ratio*100, n_dim))
    if dim is not None:
        explained_variance = pca.explained_variance_ratio_.cumsum()
        print("{} % variance was explained with {} components".format(
            explained_variance[-1]*100, dim))

    return pca


def _get_pair_indicies(X):
    """
    Determines all possible parirings of the samples in the input matrix
    Returns two vectors, where X1[i] and X2[i] together makes a pair.

    Input:
        X: Input data matrix of size [n_samples, n_features]

    Output:
        X1, X2: Indecies vectors each containg the indecies for a pair. Their dimension is equal to the binomial coefficent of the number of samples while choosing two at a time
    """

    num = X.shape[0]
    X1, X2 = np.meshgrid(np.arange(0, num), np.arange(0, num))
    X1, X2 = X1[X1 < X2], X2[X1 < X2]

    return X1, X2


def _isSymmetric(X):
    """
    Checks if the input matrix is symmetric.

    Input:
        X: Square matrix

    Output:
        Boolean stating whether the input matrix is symmetric   
    """

    rows, cols = X.shape

    if rows == cols:
        return np.allclose(X, X.T)
    else:
        print("The input matrix is not square!")
        return False


def _forceSymmetric(X):
    """
    Force the input matrix X to be symmetric by saying (X+X.T)/2

    Input:
        X: Square matrix
    Output:
        Symmetric matrix
    """

    X_half = X/2
    return X_half + X_half.T


def _force_onto_PSD_Cone(X):
    """
    Forces the input matrix onto the positiv semi-definite (PSD) cone of matrices
    Done by eigen decomposing the matrix, setting all negative eigenvalues to 0 and recompute the matrix

    NOTE: The input matrix is currently assumed to be real, and if not is forced to be real

    Input:
        X: Square matrix

    Output:
        X_psd: The input matrix forced onto the PSD cone
    """

    X = _forceReal(X)

    eig_val, eig_vec = np.linalg.eig(X)  # Get eigen values and eigenvectors
    eig_val = eig_val.real  # Force eigenvalues and eigenvectors to be real. Since we only work with covariance matrices, the only reason they should be complex is due to numerical misapproximation
    eig_vec = eig_vec.real
    eig_val[eig_val < 0] = 0  # Set all negative values to 0
    # Create diagnal matrix with the adjusted eigenvalues
    Gamma = np.diag(eig_val)

    # Reconstruct the input matrix, now as PSD
    if not _isSymmetric(X):
        print("WARNING: THE INPUT MATRIX IS NOT SYMMETRIC")
        X_psd = (eig_vec.dot(Gamma.dot(np.linalg.inv(eig_vec)))
                 ).astype(np.float64)
    else:
        X_psd = (eig_vec.dot(Gamma.dot(eig_vec.T))).astype(np.float64)

    return X_psd


def _NSDtoPSD(X):
    # TODO : Unsure whether this is mathematically sound
    """
    Determines whether the input matrix is Negative (semi-)Definite, and if so makes it Postive (semi-)Definite
    If the matrix is already Positive (semi-)Definite or Indefinite, the original matrix is returned

    It is assumed X is a real matrix, and if not it is forced to be real.

    Input:
        X: Input matrix
    Output:
        X: Corrected matrix
    """

    X = _forceReal(X)

    eig_val, eig_vec = np.linalg.eig(X)  # Get eigen values and eigenvectors
    eig_val = eig_val.real  # Force eigenvalues and eigenvectors to be real. Since we only work with covariance matrices, the only reason they should be complex is due to numerical misapproximation
    eig_vec = eig_vec.real

    if (eig_val <= 0).sum() == len(eig_val):
        print("NSD {} {}".format((eig_val <= 0).sum(), len(eig_val)))
        eig_val = -eig_val
        # Create diagnal matrix with the adjusted eigenvalues
        Gamma = np.diag(eig_val)

        if not _isSymmetric(X):
            print("WARNING: THE INPUT MATRIX IS NOT SYMMETRIC")
            X = (eig_vec.dot(Gamma.dot(np.linalg.inv(eig_vec)))).astype(np.float64)
        else:
            X = (eig_vec.dot(Gamma.dot(eig_vec.T))).astype(np.float64)

    return X


def _forceReal(X):
    """
    Force the input matrix X to be real by discarding the complex components, if any

    Input:
        X: (potential complex) matrix
    Output:
        Real matrix
    """

    if np.iscomplexobj(X):
        print("WARNING: THE INPUT MATRIX IS COMPLEX")
        X = X.real

    return X


def _safeInverse(X, lambd=0.001, verbose=False):
    """
    Tries to calculate the inverse of the supplied matrix.
    If the supplied matrix is singular it is regularized by the supplied lambda value, which is added to the diagonal of the matrix

    Input:
        X: Matrix which has to be inverted
    Output:
        X_inv: Inverted X matrix
    """

    try:
        if verbose:
            print("Condition number of {}".format(np.linalg.cond(X)))
        X_inv = np.linalg.inv(X)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            if verbose:
                print("Adding regularizer of {}".format(lambd))
                print("Condition number of {}".format(np.linalg.cond(X)))
            X = X + np.diag(np.repeat(lambd, X.shape[1]))

            if verbose:
                print("Condition number of {}".format(np.linalg.cond(X)))
            X_inv = np.linalg.inv(X)
        else:
            raise

    return X_inv


def _safeSVD(X):
    """
    Tries to calculate the SVD of the input matrix X.
    If the SVD process does not converge a boolean False is returned

    Input:
        X: Matrix which SVD is applied on
    Output:
        tuple: Tuple containing the U, S, V output of the SVD process. These are -1 if SVD did not converge
        bool: Indicating whether the SVD process converged or not
    """

    try:
        U, S, V = np.linalg.svd(X)
        return (U, S, V), True
    except np.linalg.LinAlgError as err:
        return (-1, -1, -1), False


def _reduceLabelRange(Y):
    """
    Takes a 1D numpy array of labels and converts them to start at 0 and sequentially increment

    Input:
        Y: Numpy matrix with labels with dimension [n_samples,]

    Output:
        Labels starting at 0 and increments sequentially
    """

    unique_labels = np.unique(Y)

    labels_map = {l: i for i, l in enumerate(unique_labels)}
    new_labels = np.asarray([labels_map[l] for l in Y])

    return new_labels


def _calc_Mahalanobis_distance(Xp, Xg, M, psd=False):
    """
    Calcualtes the squared Mahalanobis distance between the supplied probe and gallery images

    Input:
        Xp: Numpy matrix containg the probe image features, with dimesnions [n_probe, n_features]
        Xg: Numpy matrix containg the gallery image features, with dimesnions [n_gallery, n_features]
        M: The Mahalanobis matrix to be used, with dimensions [n_features, n_features]
        psd: Describes whether M is a PSD matrix or not. If True the sklearn pairwise_distances function will be used, while if False a manual implementation is used

    Output:
        dm: Outputs a distance matrix of size [n_probe, n_gallery]
    """

    if psd:
        return pairwise_distances(Xp, Xg, metric="mahalanobis", VI=M)
    else:
        mA = Xp.shape[0]
        mB = Xg.shape[0]
        dm = np.empty((mA, mB), dtype=np.double)
        for i in range(0, mA):
            for j in range(0, mB):
                difference = Xp[i] - Xg[j]
                dm[i, j] = difference.dot(M.dot(difference.T))
        return dm


class KISSME():
    """
    Implementation of the KISSME algorithm by Köstinger et al. ( https://files.icg.tugraz.at/seafhttp/files/1779a2d0-14b0-4046-b333-46f045959522/koestinger_cvpr_2012.pdf )

    The improved version proposed by Yang et al. is also implemented ( http://www.cbsr.ia.ac.cn/users/yyang/Yang%20Yang's%20Homepage.files/AAAI16/AAAI16_LSSL.pdf )
    """

    def __init__(self, m_name_load=None):
        # Specify M name to load it
        self.M = None
        self.psd = None
        self.base_path = '/workspace/data/metric_learning'
        if m_name_load is not None:
            self.M = self.load_m(m_name_load)

    def fit(self, X, Y, method="original", psd=True):
        """
        Code is based on https://github.com/Cysu/dgd_person_reid/blob/master/eval/metric_learning.py

        Calculates the Mahalanobis matrix M

        Input:
            X: Input data of dimensions [n_samples, n_features]
            Y: Labels of input data of dimension: [n_samples, ]
            method: Which method should be used to calculate:
                "original": Calcualtes the Mahalanobis matrix M as described in the original paper
                "improved": Calculates the Mahalanobis matrix M as described in "Large Scale Similarity Learning Using Similar Pairs for Person Verification"
            psd: Force the Mahalnobis matrix M onto the PSD cone
        """

        self.psd = psd
        M = np.eye(X.shape[1])

        # Get all possible combinations of the given datamatrix, and find the
        # number of within-class comparisons (called matches)
        X1, X2 = _get_pair_indicies(X)
        matches = (Y[X1] == Y[X2])
        num_matches = matches.sum()

        if method == "original":
            num_non_matches = len(matches) - num_matches

            # Calculate matrix of similar samples
            idxa = X1[matches]
            idxb = X2[matches]
            S = X[idxa] - X[idxb]
            C1 = S.T.dot(S) / num_matches

            # Calcualte matrix of dissimilar samples
            idxa = X1[matches == False]
            idxb = X2[matches == False]

            # Choose as many non-matches, as matches used (Done in the official Matlab code)
            p = np.random.choice(num_non_matches, num_matches, replace=False)
            idxa = idxa[p]
            idxb = idxb[p]
            S = X[idxa] - X[idxb]

            # This code replicates the MATLAB implementation, to nearly the same, but it is a bit hacky
            #p = np.random.choice(num_matches, num_matches, replace=False) + 316
            #S = X[:316] - X[p]
            C0 = S.T.dot(S) / num_matches
        elif method == "improved":
            # Uses the "improved" KISS metric proposed in the LSSL paper

            # Calculate matrix of similar samples
            idxa = X1[matches]
            idxb = X2[matches]
            S = X[idxa] - X[idxb]
            M = X[idxa] + X[idxb]

            # Calculate matches and non-matches covariance matrices
            C1 = S.T.dot(S) / num_matches
            C0 = (S.T.dot(S) + M.T.dot(M)) / (2*num_matches)
        else:
            raise ValueError("The supplied method is not supported")

        # Calculate the Mahalanobis matrix
        C0_inv = _safeInverse(C0)
        C1_inv = _safeInverse(C1)

        if not _isSymmetric(C0_inv):
            C0_inv = _forceSymmetric(C0_inv)

        if not _isSymmetric(C1_inv):
            C1_inv = _forceSymmetric(C1_inv)

        M = C1_inv - C0_inv

        M = _NSDtoPSD(M)

        if psd:
            M = _force_onto_PSD_Cone(M)

        self.M = M

    def get_distance(self, Xp, Xg=None):
        """
        Calculates the Mahalanobis distance between all probe and gallery features

        Input:
            Xp: The probe features of dimension [n_probe, n_features]
            Xg: The gallery features of dimension [n_gallery, n_features]        

        output:
            A distance matrix D of size [n_probe, n_gallery], where D[i,j] is the distance between probe i and gallery j        
        """

        assert self.M is not None, "The Mahalanobis matrix M have not been calculated. Maybe you haven't called the fit function yet?"

        # If only one input matrix, copy it
        if Xg is None:
            Xg = Xp

        return _calc_Mahalanobis_distance(Xp, Xg, self.M, False)

    def save_m(self, save_name):
        save_path = os.path.join(self.base_path, save_name + '.npy')
        if self.M is not None:
            print(self.M)
            np.save(save_path, self.M)

    def load_m(self, load_name):
        path = os.path.join(self.base_path, load_name)
        print(path + '.npy')
        M = np.load(path + '.npy', allow_pickle=True)
        return M


class XQDA():
    """
    Implementation of the "Cross-view Quadratic Discriminant Analysis" (XQDA) algorithm by Liao et al. ( https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liao_Person_Re-Identification_by_2015_CVPR_paper.pdf )
    """

    def __init__(self, lambd=0.001, dims=-1):
        self.lambd = lambd
        self.dims = dims
        self.psd = None
        self.M = None
        self.W = None

    def fit(self, X, Yx, Z=None, Yz=None, psd=False):
        """
        Code is based on the official Matlab implmentation: http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/

        Calculates a subspace where the Mahalanobis distance can be calculated

        Input:
            X: Input data of dimensions [n_samples, n_features]
            Yx: Labels of input data of dimension: [n_samples, ]
            Z: Input data of dimensions [n_samples, n_features] (Optional)
            Yz: Labels of input data of dimension: [n_samples, ] (Optional)
            psd: Force the Mahalnobis matrix M onto the PSD cone (Optional)
        """

        assert (Z is not None and Yz is not None) or (
            Z is None and Yz is None), "The Z and Yz should either both be set or none of them should be set. The received arguments are: Z = {} and Yz = {}".format(type(Z), type(Yz))

        self.psd = psd

        # If only one data/label input, just copy it
        if Z is None:
            Z = X.copy()
            Yz = Yx.copy()

        numX, n_features = X.shape
        numZ = Z.shape[0]

        # If there are more feature dimensions than total amount of samples,
        # then apply the QR Decomposition and calculate XQDA in the reduced
        # feature space
        Q = None
        if n_features > numX+numZ:
            Q, R = np.linalg.qr(np.concatenate((X.T, Z.T), axis=1))
            print(np.concatenate((X.T, Z.T), axis=1).shape)
            X = R[:, :numX].T
            Z = R[:, numX:].T
            print(Q.shape, R.shape)
            del R

        # Force the labels to be sequtential from 0
        Y = _reduceLabelRange(np.concatenate((Yx, Yz), axis=0))
        Yx = Y[:Yx.shape[0]]
        Yz = Y[Yx.shape[0]:]

        # Calculate the intra and extra covariance matrices
        # as described in Section 4.3
        classesX = np.unique(Yx)
        classesZ = np.unique(Yz)

        classes = np.unique(np.concatenate((classesX, classesZ)))
        classes = len(classes)

        class_sumX = np.zeros(shape=[classes, X.shape[1]])
        class_weightsX = np.zeros(shape=[X.shape[0], 1])

        class_sumZ = np.zeros(shape=[classes, Z.shape[1]])
        class_weightsZ = np.zeros(shape=[Z.shape[0], 1])

        # Number of simmilar pairs possible
        nI = 0

        for i in range(classes):
            classXInd = np.where(Yx == i)
            classZInd = np.where(Yz == i)

            nX = len(classXInd)
            nZ = len(classZInd)

            class_sumX[i, :] = np.sum(X[classXInd], axis=0)
            class_weightsX[np.where(Yx == i), :] = np.sqrt(nZ)

            class_sumZ[i, :] = np.sum(Z[classZInd], axis=0)
            class_weightsZ[np.where(Yz == i), :] = np.sqrt(nX)

            nI += nX * nZ

        X_tilde = class_weightsX * (X)
        Z_tilde = class_weightsZ * (Z)
        X_sum = np.sum(class_sumX, axis=0, keepdims=1)
        Z_sum = np.sum(class_sumZ, axis=0, keepdims=1)

        C1 = X_tilde.T.dot(X_tilde) + Z_tilde.T.dot(Z_tilde) - \
            class_sumX.T.dot(class_sumZ) - class_sumZ.T.dot(class_sumX)
        C0 = numZ*X.T.dot(X) + numX*Z.T.dot(Z) - \
            X_sum.T.dot(Z_sum) - Z_sum.T.dot(X_sum) - C1

        # Number of dissimilar pairs
        nE = numX*numZ - nI

        print("nE: {}, nI: {}".format(nE, nI))

        C1 /= nI
        C0 /= nE

        # Add a regularizer to the intra-personal covariance, to avoid it being
        # a singular matrix
        C1 += np.diag(np.repeat(self.lambd, X.shape[1]))

        del X_tilde, Z_tilde, X_sum, Z_sum, class_sumX, class_sumZ, class_weightsZ, class_weightsX

        # Solve the linear system C1 * x = C0
        # Solution would be equal to inv(C1).dot(C0), but we want to avoid
        # inverting the matrix
        CI1 = np.linalg.solve(C1, C0)

        # Get left singular vectors, U, and singular values, S
        # Singular values of A are equal to the sqrt of the eigenvalues of AA^T and A^TA
        # Left singular vectors are equal to the eigenvectors of AA^T
        SVD, completed = _safeSVD(CI1)
        self.SVD_completed = completed

        if completed:
            U, S, _ = SVD

            # Find the amount of singular values with a value above 1. The result is
            # the size of the new feature dimension (given it is positive and
            # lower than a requested feature dimension size)
            r = np.where(S > 1)[0].shape[0]

            if self.dims > r:
                self.dims = r

            if self.dims <= 0:
                self.dims = max(1, r)

            # Some analysis of how discriminative the chosen singular vectors/values are
            energy = np.sum(S)
            minv = S[-1]
            energy = np.sum(S[:r]) / energy
            print('Energy remained: {}, max: {}, min: {}, all min: {}, #opt-dim: {}, qda-dim: {}'.format(
                energy, S[0], S[max(0, r-1)], minv, r, self.dims))

            # Select the eigenvectors which will be used for projecting the data
            # If QR decomposition was performed, multiply the chosen eigenvectors
            # with the Q matrix
            U = U[:, :self.dims]
            if Q is None:
                self.W = U
            else:
                self.W = Q.dot(U)

            # Calculate the Mahalanobis matrix, and force it to be PSD if chosen
            C1 = U.T.dot(C1.dot(U))
            C0 = U.T.dot(C0.dot(U))

            M = np.linalg.inv(C1) - np.linalg.inv(C0)
            M = _NSDtoPSD(M)

            if psd:
                M = _force_onto_PSD_Cone(M)

            self.M = M

    def get_distance(self, Xp, Xg=None):
        """
        Calculates the Mahalanobis distance between all probe and gallery features in the found subspace

        Input:
            Xp: The probe features of dimension [n_probe, n_features]
            Xg: The gallery features of dimension [n_gallery, n_features]        

        output:
            A distance matrix D of size [n_probe, n_gallery], where D[i,j] is the distance between probe i and gallery j        
        """

        assert self.M is not None and self.W is not None, "The Mahalanobis matrix M and subspace transformation W have not been calculated. Maybe you haven't called the fit function yet?"

        # If only one input matrix, copy it
        if Xg is None:
            Xg = Xp.copy()

        # Project the probe and gallery data into the found subspace
        Xp = Xp.dot(self.W)
        Xg = Xg.dot(self.W)

        return _calc_Mahalanobis_distance(Xp, Xg, self.M, False)


class LSSL():
    """
    Implementation of the "Large Scale Similarity Learning Using Similar Pairs for Person Verification" (LSSL) algorithm by Yang et al. ( http://www.cbsr.ia.ac.cn/users/yyang/Yang%20Yang's%20Homepage.files/AAAI16/AAAI16_LSSL.pdf )
    """

    def __init__(self, lambd):
        self.lambd = lambd
        self.A = None
        self.B = None
        #self.Mb = None
        #self.Md = None

    def fit(self, X, Y, psd=False):
        """
        Calculates Simmilartiy and Dissimilarity matrices, which can be used for determining simillarty between two samples

        Input:
            X: Input data of dimensions [n_samples, n_features]
            Y: Labels of input data of dimension: [n_samples, ]
            psd: Force the output matrices onto the PSD cone
        """

        # Get all possible combinations of the given datamatrix, and find the
        # number of within-class comparisons (called matches)
        X1, X2 = _get_pair_indicies(X)
        matches = (Y[X1] == Y[X2])
        num_matches = matches.sum()

        # Calculate The dissimilary and simmilarity matrices
        idxa = X1[matches]
        idxb = X2[matches]

        # Normalize the data
        X = X / np.linalg.norm(X, ord=2, axis=1, keepdims=True)

        # Calculate commonness and difference of the similar pairs
        S = X[idxa] - X[idxb]
        M = X[idxa] + X[idxb]

        # Calulate the commonness and difference covariance matrices for
        # similar pairs, as shown in Eq. 10
        C_comm = M.T.dot(M) / num_matches
        C_diff = S.T.dot(S) / num_matches

        # Calcualte covariance matrix for dissimilar pairs as shown in Eq. 11
        C_diss = (S.T.dot(S) + M.T.dot(M)) / (2*num_matches)

        # Determine the A and B matrices as described by Eq. 12
        CI_diss = _safeInverse(C_diss)
        CI_comm = _safeInverse(C_comm)
        CI_diff = _safeInverse(C_diff)

        if not _isSymmetric(CI_diss):
            CI_diss = _forceSymmetric(CI_diss)

        if not _isSymmetric(CI_comm):
            CI_comm = _forceSymmetric(CI_comm)

        if not _isSymmetric(CI_diff):
            CI_diff = _forceSymmetric(CI_diff)

        A = CI_diss - CI_comm
        B = CI_diff - CI_diss
        A = _NSDtoPSD(A)
        B = _NSDtoPSD(B)
        # Determine Mb and Md as per Eq. 14
        #Mb = 4 * A
        #Md = CI_comm + self.lambd*CI_diff - (1+self.lambd)*CI_diss

        # Force to PSD if selected
        if psd:
            A = _force_onto_PSD_Cone(A)
            B = _force_onto_PSD_Cone(B)
            #Mb = _force_onto_PSD_Cone(Mb)
            #Md = _force_onto_PSD_Cone(Md)

        self.A = A
        self.B = B
        #self.Md = Md
        #self.Mb = Mb

    def get_similarity(self, Xp, Xg=None):
        """
        Calculates the similarity between all probe and gallery features

        Input:
            Xp: The probe features of dimension [n_probe, n_features]
            Xg: The gallery features of dimension [n_gallery, n_features]        

        output:
            dm: A similarity matrix of size [n_probe, n_gallery], where D[i,j] is the distance between probe i and gallery j        
        """

        assert (self.A is not None) and (
            self.B is not None), "The matrices A and B have not been calculated. Maybe you haven't called the fit function yet?"

        # If only one input matrix, copy it
        if Xg is None:
            Xg = Xp

        # Normalize the probe and gallery features
        Xp = Xp / np.linalg.norm(Xp, ord=2, axis=1, keepdims=True)
        Xg = Xg / np.linalg.norm(Xg, ord=2, axis=1, keepdims=True)

        # Calculate equation 5 and 13
        mA = Xp.shape[0]
        mB = Xg.shape[0]
        dm = np.empty((mA, mB), dtype=np.double)
        #dm2 = np.empty((mA, mB), dtype=np.double)
        for i in range(0, mA):
            for j in range(0, mB):
                common = Xp[i] + Xg[j]
                difference = Xp[i] - Xg[j]
                dm[i, j] = common.dot(self.A.dot(
                    common.T)) - self.lambd*difference.dot(self.B.dot(difference.T))
                #dm2[i,j] = Xp[i].dot(self.Mb.dot(Xg[j].T)) - difference.dot(self.Md.dot(difference.T))

        return dm  # , dm2


class NullSpace():
    """
    Implementation of the Discriminative Null Space algorithm by Zhang et al. ( https://arxiv.org/pdf/1603.02139.pdf )

    It is based on the Null Foley-Sammon Transform proposed by Guo et al. ( https://ieeexplore.ieee.org/document/4721915/ )    
    """

    def __init__(self, kernalized=True):
        self.W = None
        self.kernalized = kernalized

    def fit(self, X, Y):
        """
        Calculates a null subspace, collapsing datapoints from the same class onto a single point, wherein the Euclidean distance can be calculated

        Input:
            X: Input data of dimensions [n_samples, n_features]
            Y: Labels of input data of dimension: [n_samples, ]
        """

        Y = _reduceLabelRange(Y)
        classes = np.unique(Y)
        classes = len(classes)

        if not self.kernalized:
            raise ValueError(
                "The standard version of DNS is not implemented. Please use the Kernelized version with a Linear kernel")

        else:
            """
            This code is a port of the official Matlab implementation found at https://github.com/lzrobots/NullSpace_ReID            
            """

            rows, columns = X.shape
            assert rows == columns, "The supplied kernel matrix is not square!"

            N = rows
            del rows, columns

            # Center the supplied kernel Matrix X
            X_tilde = self._centerKernelMatrix(X)

            # Find eigenvectors and values of the centered kernel Matrix
            # Corresponds to find the left singular vectors and singular values of X_tilde
            # See: https://stats.stackexchange.com/a/131292
            E, V = np.linalg.eig(X_tilde)
            E = _forceReal(E)
            V = _forceReal(V)

            idx = E.argsort()[::-1]
            E = E[idx]
            V = V[:, idx]

            # Find non zero eigenvalues, and corresponding vectors.
            # There should be N-1 of these, with N being number of samples
            E_mask = ~np.isclose(E, np.zeros((E.shape[0])))

            V = V[:, E_mask]
            E = E[E_mask]

            # Scale eigenvalues by the inverse square root as stated in the paper
            # V_tilde contains the kernalized orthonormal basis (Why?)
            E = np.diag(1/(np.sqrt(E)))
            V_tilde = V.dot(E)

            del E, V

            # Create the matrices used to calculate the helper matrix H
            # Identity matrix
            I = np.eye(N, N)

            # NxN matrix with all elements set to 1/N
            M = np.ones((N, N))/N

            # TODO: Refactor this section in a vectorized manner
            # Block diagonal matrix
            L = np.zeros((N, N))
            for i in range(classes):
                class_indicies = np.where(Y == i)
                n_class = class_indicies[0].shape[0]
                C0, C1 = np.meshgrid(np.arange(0, n_class),
                                     np.arange(0, n_class))
                for i in range(C0.shape[0]):
                    for j in range(C0.shape[1]):
                        row = class_indicies[0][C0[i, j]]
                        col = class_indicies[0][C1[i, j]]
                        L[row, col] = 1/n_class

            # Calculate the helper matrix H, Eq. 12
            H = ((I-M).dot(V_tilde)).T.dot(X).dot(I-L)

            # Determine matrix which we wikll find null space of, Eq. 13
            T = H.dot(H.T)

            # Null space of T, Eq. 13
            # Should return C-1 vectors
            eigenvectors, svd_completed = self._nullSpaceBasis(T)

            self.SVD_completed = svd_completed

            if svd_completed:
                # In case T is full rank, just take the eigenvector with the smallest eigenvalue
                if eigenvectors.shape[1] < 1:
                    eigenvals, eigenvectors = np.linalg.eig(T)
                    eigenvals = _forceReal(eigenvals)
                    eigenvectors = _forceReal(eigenvectors)

                    minEigID = np.argmin(eigenvals)
                    eigenvectors = eigenvectors[:, minEigID].reshape(-1, 1)

                # Compute the Null Projecting Directions (NPDs) as per Eq. 14
                self.W = ((I-M).dot(V_tilde)).dot(eigenvectors)

    def _centerKernelMatrix(self, kernel):
        """
        Calculates the centered Kernel Matrix as per https://github.com/lzrobots/NullSpace_ReID/blob/master/DNS.m
        Is equal to the standard (I-1_n)K(I-1_n) matrix centering equation ( see: https://stats.stackexchange.com/a/131292 )
        Where:
            I is the identity matrix
            K is the kernel matrix
            1_n is a matrix with all values set to 1/n with n = number of data points

        Input:
            kernel: The kernel matrix which has to be centered. Has to be square

        Output:
            centeredKernel: The centered input kernel, which is also square
        """

        assert _isSymmetric(
            kernel), "The supplied kernel matrix is not symmetric!"

        dim = kernel.shape[0]

        # NOTE: columnMeans = rowMeans because kernelMatrix is symmetric
        columnMeans = np.mean(kernel, axis=0)
        matrixMean = np.mean(columnMeans)

        centeredKernel = kernel.copy()

        for k in range(dim):
            centeredKernel[k, :] -= columnMeans
            centeredKernel[:, k] -= columnMeans

        centeredKernel += matrixMean

        return centeredKernel

    def _nullSpaceBasis(self, A, atol=1e-13, rtol=0):
        """
        Calculates the null space basis of the input matrix, i.e. it returns the orthonormal basis that corresponds to 0 values eigenvalues/singular values
        SVD is used and right hand singular vectors corresponding to 0-valued singular values are chosen

        The code is based on: http://scipy-cookbook.readthedocs.io/items/RankNullspace.html#rank-and-nullspace-of-a-matrix


        Input:
            A: Input matrix, which should be square
            atol : The absolute tolerance for a zero singular value. Singular values smaller than `atol` are considered to be zero.
            rtol : The relative tolerance.  Singular values less than rtol*smax are considered to be zero, where smax is the largest singular value.

        Output:
            eig_vecs: The eigen vectors of the null space of A
        """

        SVD, completed = _safeSVD(A)

        if completed:
            _, S, V = SVD
            tol = max(atol, rtol * S[0])
            rank = (S >= tol).sum()
            return V[rank:].conj().T, completed
        else:
            return None, completed

    def _gram_schmidt(self, V):
        """
        Calculates the Gram-Schmidt orthononormalization of the supplied matrix

        Uses the Modified Gram-Schmidt orthonormalization approach, as described at https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability

        Based on code by JJGO ( https://gist.github.com/iizukak/1287876#gistcomment-1871542 )


        Input:
            V: Input matrix of size [n_samples, n_features]

        Output:
            Returns an array of size [K, n_features], where K is the number of non-zero eigenvectors found

        """

        basis = []
        zeros = np.zeros(V[0].shape)

        for v in V:
            w = v
            for b in basis:
                w = w - np.dot(b, w)/np.dot(b, b) * b
            if not np.allclose(w, zeros):
                basis.append(w/np.linalg.norm(w))
        return np.array(basis)

    def get_distance(self, Xp, Xg=None):
        """
        Calculates the euclidean distance between all probe and gallery features in the found null space

        Input:
            Xp: The probe features of dimension [n_probe, n_features]
            Xg: The gallery features of dimension [n_gallery, n_features]        

        output:
            A distance matrix D of size [n_probe, n_gallery], where D[i,j] is the distance between probe i and gallery j        
        """

        assert self.W is not None, "The Null Space projection matrix W have not been calculated. Maybe you haven't called the fit function yet?"

        # If only one inpu matrix, copy it
        if Xg is None:
            Xg = Xp

        # Project the probe and gallery data into the found null space
        null_Xp = Xp.dot(self.W)
        null_Xg = Xg.dot(self.W)

        return pairwise_distances(null_Xp, null_Xg, metric="euclidean")


def RBF_kernel(X, Y=None, mu=None):
    """
    Applies the Radial Basis Function Kernel on the input data as per https://github.com/lzrobots/NullSpace_ReID/blob/master/RBF_kernel.m
        k(x,y) = exp (-0.5/mu^2 ||x-y||^2 )

    Input:
        X: Input matrix of size [n_X, n_features]
        Y: Input matrix of size [n_Y, n_features]. (Used for finding the kernel matrix of the test data)
        mu: The parameter used in the RBF kernel function. Default is the mean of the distances (Use the mu from the training data, if possible)

    Output:
        K: The kernal matrix of the input data X and Y of size [n_Y, n_X]
        mu: The parameter for the kernel
    """

    if Y is None:
        Y = X

    # Find the squared euclidean distance
    dm = pairwise_distances(X, Y, metric="euclidean")**2

    # Calculate the mu parameter as the sqrt of the mean distance / 2
    if mu is None:
        mu = np.sqrt(np.mean(dm)/2)

    return np.exp(-0.5/(mu**2) * dm).T, mu


def Linear_kernel(X, Y=None, c=0):
    """
    Applies the Linear Kernel on the input data:
        k(x,y) = x^Ty + c

    Input:
        X: Input matrix of size [n_X, n_features]
        Y: Input matrix of size [n_Y, n_features]. (Used for finding the kernel matrix of the test data)
        c: An optional constant that can be added. Default is 0

    Output:
        K: The kernal matrix of the input data X and Y of size [n_Y, n_X]
    """

    if Y is None:
        Y = X

    return Y.dot(X.T) + c
