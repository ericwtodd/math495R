# markov_chains.py
"""Volume II: Markov Chains.
Eric Todd
Math 321 - 002
October 31, 2018
"""

import numpy as np
from scipy import linalg as la


# Problem 1
def random_chain(n):
    """Creates and returns a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    # Populate a random matrix of size n.
    transition = np.random.random((n, n))
    # Divide each column by the column sum, effectively making the column sum 1, and thus a random Markov chain
    transition /= transition.sum(axis=0)
    return transition


# Problem 2
def forecast(days):
    """Forecasts the weather for a certain number of days,
     With the assumption that the first day is hot.

    Parameter:
        days (int): The number of days to forecast the weather for.
    """
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    # We will forecast weather for n days
    n = days

    # Create a list containing the predicted forecasts
    predictions = []

    while n > 0:
        # We assume the first day is hot, and we check whether the previous day was hot
        if n == days or predictions[-1] == 0:
            predictions.append(np.random.binomial(1, transition[1, 0]))
        # If the last day was cold, choose from the probability that it is cold the day before
        else:
            predictions.append(np.random.binomial(1, transition[1, 1]))
        n -= 1
    return predictions


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    # Create Out Markov Transition Matrix
    four_states = np.array([[0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]])
    # Create a list to hold our predictions
    predictions = []
    # Get predictions for n = days
    n = days

    while n > 0:
        # We treat the first day as mild, and then check if the previous day was mild
        # To use the mild probabilities to find the next state
        if n == days or predictions[-1] == 1:
            value = np.argmax(np.random.multinomial(1, four_states[:, 1]))
            predictions.append(value)
        # We then check if the previous day was hot, to use it's probabilities to get the next state
        elif predictions[-1] == 0:
            value = np.argmax(np.random.multinomial(1, four_states[:, 0]))
            predictions.append(value)
        # If the previous day was cold we use those probabilities to get the next state
        elif predictions[-1] == 2:
            value = np.argmax(np.random.multinomial(1, four_states[:, 2]))
            predictions.append(value)
        # If the previous day was freezing we use those probabilities to get the next state
        elif predictions[-1] == 3:
            value = np.argmax(np.random.multinomial(1, four_states[:, 3]))
            predictions.append(value)
        n -= 1
    return predictions


# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Computes the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """

    # Get the shape of A
    m, n = A.shape
    # Generate a random state distribution vector x0
    x0 = np.random.random(n)
    x0 /= sum(x0)

    # Start at k = 0
    k = 0
    # Now we calculate x_k+1
    while True:
        # If k exceeds N, raise a ValueError
        if k > N:
            raise ValueError("A^k did not converge")
        else:
            # Calculate x_k+1 (xk) as A @ x_k (A @ x0)
            xk = A @ x0
            # If the norm is less than the convergence tolerance, we have convergence
            # So we exit the loop
            if la.norm(xk - x0) < tol:
                break
            # Otherwise, we set our x_k as our new x0, and iterate again
            x0 = xk
            # Increment k to model the current k of A^k
            k += 1
    return xk


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        filename (str) : the name of the file read in
        states (list) : a list containing each unique state in the input file
        transition ((n+2,n+2), ndarray):
        size (int): the size of the transition matrix (n+2 x n+2)
        contents (list)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """

    def __init__(self, filename):
        """Reads the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        with open(filename, 'r') as in_file:
            # Store the name of the file
            self.filename = filename
            # Calculate the number of unique words and add two to get the size
            # n+2 of the transition matrix
            all_states = list(in_file.read().split())
            # Add the stop and start states to all states, and get the size
            all_states.insert(0, "$tart")
            all_states.append("$top")
            self.size = len(np.unique(all_states))
            # Generate an initialized Transition matrix of zeros of size n+2 x n+2
            self.transition = np.zeros((self.size, self.size))
            # Go back to the beginning of a file, and get each line/sentence
            in_file.seek(0)
            self.contents = in_file.read().split("\n")

            # Initialize a list of states
            self.states = ["$tart"]

            # Going through each sentence, we populate our transition matrix
            for line in self.contents:
                words = line.split()
                for word in words:
                    # Append each new word to the list of states
                    if word not in self.states:
                        self.states.append(word)
                # Get the indices of each word in the sentence
                word_indices = [self.states.index(word) for word in words]
                # If there is an extra hanging line, with no words, we want to account for that
                if len(word_indices) != 0:
                    # Add one to the entry of the transition matrix corresponding
                    # to going from the start state to the first word of the sentence
                    self.transition[word_indices[0], 0] += 1
                    # Add the relations of the words in the sentence
                    for i in range(len(word_indices) - 1):
                        self.transition[word_indices[i + 1], word_indices[i]] += 1
                    # Add the transition from the last word to the stop state
                    self.transition[self.size - 1, word_indices[-1]] += 1

            # Add stop to the list of states
            self.states.append("$top")
            # Make sure the stop state transitions to itself.
            self.transition[self.size - 1, self.size - 1] += 1
            # Normalize the column by dividing by the column sums
            self.transition /= self.transition.sum(axis=0)

    def babble(self):
        """Begins at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        # Start at the start state
        current_state = 0
        sentence = []
        # While we haven't transitioned to the stop state,
        # we add words to our sentence
        while True:
            # Get an array that has a 1 in the index where the word was chosen, 0's everywhere else.
            next_word_chosen = np.random.multinomial(1, self.transition[:, current_state])
            # Get the index of the word chosen, which is the new current state
            current_state = int(np.argmax(next_word_chosen))
            # If we are at $top, we break, otherwise, we add the word to our sentence and regenerate another word based on the current state
            if current_state == self.size - 1:
                break
            else:
                sentence.append(self.states[current_state])
        # Return the sentence as a string
        return " ".join(sentence)
