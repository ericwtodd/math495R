# image_segmentation.py
"""Volume 1: Image Segmentation.
Eric Todd
Math 345 - 002
November 1, 2018
"""

import numpy as np
from scipy import linalg as la
from imageio import imread
from imageio import imwrite
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla


# Problem 1
def laplacian(A):
    """Computes the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    # Compute the Degree Matrix D, of G using A
    # Where D's ith diagonal is the sum of the ith column of A
    D = np.diag(A.sum(axis=0))
    # The Laplacian is L = D-A, so we return it
    return D - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Computes the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # Get our Laplacian Matrix
    L = laplacian(A)
    # Compute the Eigenvalues of the laplacian matrix L
    eigenvalues = list(np.real(la.eigvals(L)))
    # Treat values less than the tolerance as eigenvalues of 0
    # And find all the eigenvalues = 0
    components = 0
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < tol:
            eigenvalues[i] = 0
            components += 1
    # Find the smallest eigenvalue and remove it in order to look at the 2nd smallest one
    smallest = min(eigenvalues)
    eigenvalues.remove(smallest)
    # The algebraic connectivity is the value of the second smallest eigenvalue
    alg_connectivity = min(eigenvalues)
    return components, alg_connectivity


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculates the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col) ** 2 + (Y - row) ** 2))
    mask = R < radius
    return (X[mask] + Y[mask] * width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.file_name = filename
        # Store the original image
        self.image = imread(filename)
        # Scale the image so it's values are between 0 and 1
        self.scaled_image = self.image / 255
        # If the image is in color, then we store the brightness matrix
        if len(self.scaled_image.shape) == 3:
            self.brightness = self.scaled_image.mean(axis=2)
        # Otherwise the brightness matrix is just the scaled image itself
        else:
            self.brightness = self.scaled_image
        # Also store the flattened brightness array (1-D) as an attribute
        self.flattened = np.ravel(self.brightness)

    # Problem 3
    def show_original(self):
        """Displays the original image."""
        # If the original image was color, we display normally without a specific color map
        if len(self.scaled_image.shape) == 3:
            plt.imshow(self.scaled_image)
            plt.axis("off")
        # Otherwise, if it is black & white, and we need to specify the colormap as gray
        # And plot it
        else:
            plt.imshow(self.scaled_image, cmap="gray")
            plt.axis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Computes the Adjacency and Degree matrices for the image graph."""
        # Gets the size of the original image
        m, n = self.scaled_image.shape[:2]
        # Initialize A as a (mn) x (mn) matrix
        A = sparse.lil_matrix((m * n, m * n))
        # initialize D as an mn-array of 0's
        D = np.zeros((m * n))
        # We go through each value in D and add the weights of its neighbors in it
        for i in range(m * n):
            # Find the neighbors for each pixel
            neighbors, distances = get_neighbors(i, r, m, n)
            weights = []
            # Compute the weights for each neighbor of the pixel
            for j in range(len(distances)):
                if distances[j] < r:
                    weights.append(
                        np.exp((-1 * abs(self.flattened[i] - self.flattened[neighbors[j]]) / sigma_B2 - distances[
                            j] / sigma_X2)))
                else:
                    weights.append(0)
            # Add the weights into the A matrix in the appropriate places, other entries are 0
            # since its a sparse matrix
            A[i, neighbors] = weights
            # Get the row sum and put that as the ith diagonal entry of D
            D[i] = A[i].sum(axis=1)
            # Return A as a csc matrix, and D, as an array
        return A.tocsc(), D

    # Problem 5
    def cut(self, A, D):
        """Computes the boolean mask that segments the image."""
        # Compute the Laplacian matrix L, of A.
        L = sparse.csgraph.laplacian(A)
        # Compute D ^ -1/2
        degree = sparse.diags(D ** (-0.5))
        # Compute the two smallest eigenvalues and their corresponding eigenvectors
        evalues, evectors = spla.eigsh(degree @ L @ degree, which="SM", k=2, return_eigenvectors=True)
        # Get the size of the original image
        m, n = self.scaled_image.shape[:2]
        # Chooses the second smallest eigenvalue's eigenvector
        # to use as the mask that will separate the two segments of the image
        if evalues[0] > evalues[1]:
            #Make the vector consistently return the positive and then the negative
            if evectors[0,0] > 0:
                evectors *= -1
            mask = evectors[:, 0].reshape((m, n)) > 0
        else:
            #Make the vector consistently return the positive and then the negative
            if evectors[0,1] > 0:
                evectors *= -1
            mask = evectors[:, 1].reshape((m, n)) > 0
        # print(mask)
        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)
        segment_mask = self.cut(A, D)

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        # If the image is in color, we stack three masks for each RGB triplet, and plot the
        # images in color
        if len(self.scaled_image.shape) == 3:
            # Create the positive image using the mask
            positive_image = self.scaled_image * np.dstack((segment_mask, segment_mask, segment_mask))
            # Create the negative image using the negation of the mask
            negative_image = self.scaled_image * np.dstack((~segment_mask, ~segment_mask, ~segment_mask))
            ax1.imshow(self.scaled_image)
            ax1.axis("off")
            ax2.imshow(positive_image)
            ax2.axis("off")
            ax3.imshow(negative_image)
            ax3.axis("off")
        # Otherwise if the image is not in color, we only need to use one mask
        # as well as the gray color map
        else:
            # Create the positive image using the mask
            positive_image = self.scaled_image * segment_mask
            # create the negative image using the negation of the mask
            negative_image = self.scaled_image * (~segment_mask)
            # Plot the images in grayscale
            ax1.imshow(self.scaled_image, cmap="gray")
            ax1.axis("off")
            ax2.imshow(positive_image, cmap="gray")
            ax2.axis("off")
            ax3.imshow(negative_image, cmap="gray")
            ax3.axis("off")
        plt.show()

# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
