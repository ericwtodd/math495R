import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
import scipy.stats
from matplotlib import pyplot as plt


def exhaustive_search(X, z):
    """Solves the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    # Use array broadcasting to return the norms for each row
    norms_vector = la.norm(X - z, axis=1)
    # Return the nearest neighbor in X (using the index of the minimum distance) and its value
    return X[np.argmin(norms_vector)], min(norms_vector)


class KDTNode:
    """A node used in a k-dimensional binary tree that has a k-dimensional vector (value)
    and pointers to its left and right children, as well as a pivot index that is determined when
    the node is inserted into the tree"""

    def __init__(self, x):
        """
        Constructor for a KDTNode. Sets value to the k-dimensional array
        and sets left child, right child to None. Pivot is also set to None.

            Parameters:
                  x (np.ndarray):
        """
        # If the value passed in is not a numpy array, raise an error
        if type(x) != np.ndarray:
            raise TypeError("x must be a NumPy array")
        else:
            # Fill our node's value with x, initialize children as None
            self.value = x
            self.left = None
            self.right = None
            # The pivot will be changed when the node is inserted into the tree
            self.pivot = None

class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """

    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """

        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:  # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current  # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)  # Recursively search left.
            else:
                return _step(current.right)  # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Inserts a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """

        def _insert_at(current_node):
            """
            Recursively searches through the tree to find the correct
            location to insert the data at.
            If the data is inserted, then it stops
            """
            # If the data is already in the tree, raise a ValueError
            if np.allclose(current_node.value, data):
                raise ValueError(str(data) + " is already in the tree")
            # If the value at the data's pivot is greater than or equal to the value at the parent's pivot
            # We see if there is a child to the right. If not, we add it, otherwise, we recurse on the right child
            if data[current_node.pivot] >= current_node.value[current_node.pivot]:
                if current_node.right is None:
                    current_node.right = KDTNode(data)
                    # If the parent's pivot is k-1, we reset the child's pivot to 0,
                    if current_node.pivot == (self.k - 1):
                        current_node.right.pivot = 0
                    else:
                        # Otherwise increment the child's pivot to the parents + 1
                        current_node.right.pivot = current_node.pivot + 1
                else:
                    # If the node's right child is not none, we recurse on the right child
                    _insert_at(current_node.right)
            elif data[current_node.pivot] < current_node.value[current_node.pivot]:
                if current_node.left is None:
                    current_node.left = KDTNode(data)
                    # If the parent's pivot is k-1, we reset the child's pivot to 0,
                    if current_node.pivot == (self.k - 1):
                        current_node.left.pivot = 0
                    else:
                        # Otherwise increment the child's pivot to the parents + 1
                        current_node.left.pivot = current_node.pivot + 1
                else:
                    # If the child is not none, we recurse on the left child
                    _insert_at(current_node.left)

        # If the tree is empty, we insert at the root, and set pivot to 0
        if self.root is None:
            self.root = KDTNode(data)
            self.root.pivot = 0
            self.k = len(data)

        # If the length of the data to be inserted is not in R^k, raise an error
        elif len(data) != self.k:
            raise ValueError("Length of data does not match k: " + str(self.k))

        else:
            # Otherwise, we call our recursive function starting at the root to insert the data appropriately
            _insert_at(self.root)

    def query(self, z):
        """Finds the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """

        def kd_search(current, nearest, nearest_distance):
            """Recursively searches for the nearest neighbor to the target node.
            If the current node being looked at is None, we return the nearest node
            and it's distance from the target.
            Otherwise, it examines the distance between the current node's value
            and the target value, and updates min distance accordingly.
            Then it searches along the children of the current node expected to be closer
            to see if any are closer, as well as it's other branch if necessary.
            Parameters:
                 current(KDTNode): The current node being examined, compared to the target
                 nearest (KDTNode): The node that is currently the nearest to the target
                 nearest_distance (float): The distance of the closest node to the target
            """

            # If the current node is None, there is no value to compare to the target,
            # So we return the nearest node before we started, and the nearest distance.
            if current is None:
                return nearest, nearest_distance
            # Otherwise, we set the current node's value as x and pivot as i for convenience.
            x = current.value
            i = current.pivot  # the pivot index

            # If the current node is closer to the target than the current nearest node,
            # We update the current nearest node, and its distance to the target
            if la.norm(x - z) < nearest_distance:
                nearest = current
                nearest_distance = la.norm(x - z)
            # If the target's value at the pivot index is less than the current node's value at the pivot index,
            # Then we call our nearest neighbor search on the left subtree of the current node.
            if z[i] < x[i]:
                nearest, nearest_distance = kd_search(current.left, nearest, nearest_distance)
                # If the current node's value at the pivot index is within
                # the radius of target's value at the pivot index  + the current nearest distance,
                # We must also search the right subtree of the current node.
                if z[i] + nearest_distance >= x[i]:
                    nearest, nearest_distance = kd_search(current.right, nearest, nearest_distance)

            # Otherwise the target's value at the pivot index is greater than the current node's value at the pivot index,
            # So we call our nearest neighbor search on the right subtree of the current node.
            else:
                nearest, nearest_distance = kd_search(current.right, nearest, nearest_distance)
                # If the current node's value at the pivot index is within
                # the radius of target's value at the pivot index  + the current nearest distance,
                # We must also search the left subtree of the current node.
                if z[i] + nearest_distance >= x[i]:
                    nearest, nearest_distance = kd_search(current.left, nearest, nearest_distance)

            return nearest, nearest_distance

        # Starting at the root, we find the nearest neighbor to the target
        nearest_neighbor, closest_distance = kd_search(self.root, self.root, la.norm(self.root.value - z))
        return nearest_neighbor.value, closest_distance

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


class KNeighborsClassifier:
    """A Classifier that uses the labels of the k nearest neighbors of a target to label it
    """

    def __init__(self, n_neighbors):
        """A constructor that takes the number of neighbors that will be used to
        vote and classify a target

            Parameters:
                n_neighbors (int): The number of neighbors close to the target to be used to classify
        """
        # Set the value of n_neighbors to n_neighbors
        self.n_neighbors = n_neighbors
        # Set our tree and labels to none. They will be initialized in fit.
        self.tree = None
        self.labels = None

    def fit(self, X, y):
        """Takes the mxk training set X and an array of m labels, y,
        Loads a SciPy KDTree with the data in X, and saves the tree and labels
        as attributes of the Classifier

        Parameters:
            X (np.ndarray): an mxk array, where each row represents a k-dimensional vector
            y (np.ndarray): an array of size m, representing the labels for each row in the array X
        """
        # Store the KDTree and labels as attributes in the classifier
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """Queries the Classifier's KDTree for the k (self.n_neighbors) nearest
        neighbors to the target z, and returns the most common label of those neighbors.
        If there is a tie for the most common label, it returns the alphanumerically smallest label
        """
        # Find the k (self.n_neighbors) nearest neighbors of the target z
        distances, indices = self.tree.query(z, k=self.n_neighbors)
        # Find the most common label of the k nearest neighbors and return it
        prediction = scipy.stats.mode(self.labels[indices])
        return int(prediction.mode)


def prob6(n_neighbors, filename="mnist_subset.npz", display=False):
    """Extracts the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Uses the classifier to
    predict labels for the test data. Returns the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.
        display (bool): whether or not the first 60 values should be displayed in a graph.

    Returns:
        (float): the classification accuracy.
    """

    # Load in the data file
    data = np.load(filename)

    # Separate the training and test data by assigning them variables
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]

    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]

    # Create a KNeighborsClassifier
    my_classifier = KNeighborsClassifier(n_neighbors)
    # Fit the Classifier model with the training data X_train and training labels y_train
    my_classifier.fit(X_train, y_train)

    # Predict the labels of each image in X_test
    predictions = []
    for x in X_test:
        predictions.append(my_classifier.predict(x))
    # Cast it as a NumPy array
    predictions = np.array(predictions)

    if display:
        for i in range(60):
            plt.subplot(5, 12, i + 1)
            plt.imshow(X_test[i].reshape((28, 28)), cmap="gray")
            plt.title(predictions[i])
            plt.axis("off")
        plt.show()

    # Find the accuracy and return it
    accuracy = sum(predictions == y_test) / len(y_test)

    return accuracy
