import random
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1713)
plt.axis('equal')


def MAT_step(matrix):
    """
    Function that given a matrix in input generates a new matrices applying the
    step function for each i-j value. The step function is defined as
    - step(x) = 1 for x >= 0
    - step(x) = 0 for x < 0

    :param matrix: Input matrix
    :return: step(x)
    """
    # Initialize new temporary matrix for the result
    result = []

    # Rows
    for row in matrix:
        new_row = []
        # Element of the row
        for element in row:
            # Step function
            if element < 0:
                new_row.append(0)
            else:
                new_row.append(1)
        # Add the new row
        result.append(new_row)

    # return the new matrix
    return result


def stepFunct(val):
    """
    Function that given in input a value, evaluates it under the
    step function and returns the new value
    :param val: input value
    :return: 1 if val > 0, 0 otherwise
    """
    if val < 0:
        return 0
    else:
        return 1


def MAT_transp(A):
    """
    Function that given a vector, computes the transponse of the vector
    :param A: A list representing a vector (either row or column).
    :return: The transpose of the vector.
    """
    # Check if A is a row vector or a column vector
    if len(A) == 0:  # If the vector is empty, return an empty vector
        return []
    elif isinstance(A[0], list):  # If the first element is a list, it's a column vector
        # Convert a column vector to a row vector
        return [element[0] for element in A]
    else:  # Otherwise, it's a row vector
        # Convert a row vector to a column vector
        return [[element] for element in A]


def VECT_prod(A, B):
    """
    Function that given two vectors, computes the product of the vectors
    :param A: vector 1
    :param B: vector 2
    :return: A*B
    """
    lenA = len(A)  # get A dimension
    res = 0

    # check B has lenA rows with one element only per each row
    if len(B) == lenA:
        for i in range(len(A)):
            if len(B[i]) != 1:
                return -1 # return error value

    # product
    for i in range(len(A)):
        res += A[i] * B[i][0]

    return res


def plotPlane(x, y, weights, vector=None, point_on_line=0.5):
    """
    Plots the points x (x1, x2) in red if y[i] = 1 and in blue otherwise,
    and plots the decision boundary and optional vector.
    :param x: A list of vectors, where each vector is of the form [x0, x1, x2].
    :param y: A list of labels, each being 0 or 1.
    :param weights: A list of weights for the decision boundary.
    :param vector: An optional list [w1, w2] representing a vector to be plotted.
    :param point_on_line: A float representing a point on the decision boundary (0 to 1).
    """
    # Extract x1 and x2 from the vector x for plotting
    x1 = [point[1] for point in x]
    x2 = [point[2] for point in x]

    # Plot points in red or blue based on y
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x1[i], x2[i], color='red',
                        label='Class 1' if 'Class 1' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(x1[i], x2[i], color='blue',
                        label='Class 0' if 'Class 0' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot the decision boundary (assuming weights = [w0, w1, w2])
    x1_vals = np.linspace(min(x1), max(x1), 100)
    x2_vals = -(weights[0] + weights[1] * x1_vals) / weights[2]
    plt.plot(x1_vals, x2_vals, 'k-', label='Decision Boundary')

    # Plot the vector if provided
    if vector is not None:
        # Calculate the start point of the vector on the decision boundary
        start_x1 = min(x1) + point_on_line * (max(x1) - min(x1))
        start_x2 = -(weights[0] + weights[1] * start_x1) / weights[2]

        # Plot the vector from the chosen point
        plt.quiver(start_x1, start_x2, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='green',
                   label='Vector [w1, w2]')

    # Add labels and legend
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Scatter Plot of Points with Decision Boundary and Vector')
    plt.grid(True)
    plt.show()


def vectorRandomGen(size, boundaries):
    """
    Function that given the size of the vector and the boundaries,
    generates a random vector. If the vector has n values,
    size has to be equal to n. Moreover, the vector for the boundaries
    has the ranges for each element of the vector. Therefore,
    boundaries[i*2] will be the lower bound of the vector[i], and boundaries[i*2+1]
    will be the upper bound of the vector[i]. Thus, boundaries must be double in size
    respect to the vector size
    :param size: Number of vector elements
    :param boundaries: lower and upper bound of the elements of the vector
    :return: random vector, full of -1 if boundaries size doesn't match the vector size
    """
    vector = [-1] * size  # init vector with -1

    # check if all boundaries have been provided
    if len(boundaries) < size*2:
        # miss some boundaries
        return vector  # return vector full of -1

    # random generation
    for i in range(size):
        vector[i] = random.uniform(boundaries[i*2], boundaries[i*2+1])  # uniform random generation

    return vector  # return random vector


def main():
    # number of samples
    n = 100

    # weight vector size
    n_weights = 3

    # NN input and output
    y = []
    x = [] # vector of pairs (x0, x1, x2)

    # boundaries for w0, w1, w2, respectively
    boundaries_weights = [-0.25, 0.25, -1, 1, -1, 1]

    # boundaries for x0, x1, x2, respectively
    boundaries_x = [1, 1, -1, 1, -1, 1]

    # random generation of weight vector
    weights = vectorRandomGen(n_weights, boundaries_weights)

    # print weights
    print("Weights =", [f"{w:.2f}" for w in weights])

    # generate n samples with random x vector
    for i in range(n):
        x.append(vectorRandomGen(3, boundaries_x))  # vector [x0, x1, x2] generation

    # fill y given x, W
    for i in range(n):
        y.append(stepFunct(VECT_prod(weights, MAT_transp(x[i]))))

    plotPlane(x, y, weights, vector=[weights[1], weights[2]])

    print("x = ", x)
    print("y = ", y)
    print("weights = ", weights)


main()