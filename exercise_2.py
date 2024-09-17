import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
plt.axis('equal')
np.random.seed(1713)


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


def VECT_sum_sub(A, B, sum=1):
    """
    Function that given two vectors, computes either the sum or the sub
    based on the sum parameter. By default, the sum is computed
    :param A: Vector 1
    :param B: Vector 2
    :param sum: control parameter for selecting either sum or sub
    :return: return the sum or the sub
    """
    ret = []

    for i in range(len(A)):
        if (sum == 1):
            # sum
            ret.append(A[i] + B[i])
        else:
            # sub
            ret.append(A[i] - B[i])

    # return the result
    return ret


def VECT_prodByScalar(A, scalar):
    """
    Function that given two vectors, computes the product between one vector
    and one scalar
    :param A: Vector
    :param scalar: Scalar
    :return: return scalar*A
    """
    ret = []
    for val in A:
        # multiply each element of the vector by the scalar
        ret.append(val * scalar)
    # return the result
    return ret


def perceptronLearingAlgo(x, y, eta = 1.0, initWrandom=False, initWval = 1):
    """
    Function that given a training data set, supervised, computes a linear separator (boundary) for
    the class classification
    :param x: x data set
    :param y: y data set
    :param eta: eta parameter
    :param initWrandom: choose if start with either random weight vector or
    weight vector full of initWval (False by default)
    :param initWval: If the initialization of weight vector is not random,
    it is fulfilled by this value (1 by default)
    :return: Return the weight vector and the number of errors per epoch
    """
    # weight vector size
    weight_size = len(x[0])

    # condition while loop
    end = False

    # epochs
    epoch = []
    epoch_index = -1

    # init weight vector
    if initWrandom:
        # init with random values
        weights = [random.uniform(-1, 1) for _ in range(weight_size)]
    else:
        # init with all 1's
        weights = [initWval] * weight_size

    # check if error occurs
    while not end:
        end = True
        epoch.append(0)
        epoch_index += 1
        for i in range(len(x)):
            y_NNw = stepFunct(VECT_prod(weights, MAT_transp(x[i])))

            if y_NNw != y[i]:
                if y[i] - y_NNw < 0:
                    isSum = 0
                else:
                    isSum = 1

                weights = VECT_sum_sub(weights, VECT_prodByScalar(x[i], eta), isSum)
                end = False
                epoch[epoch_index] += 1


    return weights, epoch


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


def plotEpoch(y, eta):
    """
    Function that plots the epoch
    :param y: epochs
    :param eta: eta parameter
    :return:
    """
    plt.figure(figsize=(8, 5))
    plt.plot(y, linestyle='-', color='r', label='Errors per epoch')
    plt.title('Epochs with eta = ' + str(eta))
    plt.xlabel('epoch')
    plt.ylabel('Errors')
    plt.grid(True)
    plt.legend()
    plt.show()


def AVG_epoch(epochs):
    """
    Function that given a dictionary of epochs, returns the average epoch
    :param epochs: dictionary of epochs
    :return: dictionary of average epoch per each key
    """
    res = {}
    for k in epochs.keys():
        # AVG
        res[k] =statistics.mean(epochs[k])

    return res


def PERC_10_90_epoch(epochs):
    """
    Function that given a dictionary of epochs, returns the 10th and 90th
    percentile
    :param epochs: dictionary of epochs
    :return: both the 10th and 90th percentile stored into two dictionaries respectively
    """
    res10 = {}
    res90 = {}
    for k in epochs.keys():
        # 10th percentile
        res10[k] = float(np.percentile(epochs[k], 10))
        # 90th percentile
        res90[k] = float(np.percentile(epochs[k], 90))

    return res10, res90


def plotPlane(x, y, weights, weight_old=None, vector=None, point_on_line=0.5):
    """
    Plots the points x (x1, x2) in red if y[i] = 1 and in blue otherwise, and plots the decision boundary,
    an old decision boundary, and an optional vector.

    :param x: A list of vectors, where each vector is of the form [x0, x1, x2].
    :param y: A list of labels, each being 0 or 1.
    :param weights: A list of weights for the decision boundary.
    :param weight_old: An optional list of weights for the old decision boundary.
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

    # Plot the old decision boundary if provided (assuming weight_old = [w0_old, w1_old, w2_old])
    if weight_old is not None:
        x2_vals_old = -(weight_old[0] + weight_old[1] * x1_vals) / weight_old[2]
        plt.plot(x1_vals, x2_vals_old, 'g--', label='Old Decision Boundary')

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
    plt.title('Scatter Plot of Points with Decision Boundary, Old Decision Boundary, and Vector')
    plt.grid(True)
    plt.show()


def plotAVG(dictAVG, dictPERC10, dictPERC90, eta):
    """
    Plots the average error values for each epoch from a dictionary, adding a fill area
    for the 10th and 90th percentiles.

    Parameters:
    dictAVG (dict): Dictionary containing epochs as keys ('EP0', 'EP1', ...) and the average error as values.
    dictPERC10 (dict): Dictionary containing the 10th percentile of errors for each epoch.
    dictPERC90 (dict): Dictionary containing the 90th percentile of errors for each epoch.
    """
    # Extracts epochs (keys) and the average error values (values) from the dictionary
    epochs = list(dictAVG.keys())
    avg_errors = list(dictAVG.values())

    # Extracts percentile values from the dictionaries
    perc10 = list(dictPERC10.values())
    perc90 = list(dictPERC90.values())

    # Creates the plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_errors, linestyle='-', color='b', label='Average Errors per Epoch')

    # Adds the fill area between the 10th and 90th percentiles
    plt.fill_between(epochs, perc10, perc90, color='red', alpha=0.3, label='10th-90th Percentile Range')

    plt.title('Average Errors per Epoch with Percentiles with eta = ' + str(eta))
    plt.xlabel('Epoch')
    plt.ylabel('Average Error')
    plt.grid(True)

    # Sets x-axis labels every 5 epochs and always includes the last epoch
    xticks_positions = list(range(0, len(epochs), 5))  # Every 5 epochs
    if xticks_positions[-1] != len(epochs) - 1:
        xticks_positions.append(len(epochs) - 1)  # Adds the last epoch if it's not already included

    plt.xticks(xticks_positions, [epochs[i] for i in xticks_positions])  # Applies custom labels

    plt.legend()
    plt.show()


def main():
    # data set
    x = [[1.0, 0.28898251017186216, -0.587125706238274], [1.0, -0.2997243130770335, 0.5629936146536132],
         [1.0, 0.43415123763045815, -0.5505617706434638], [1.0, -0.9458382816040334, -0.06562181331910022],
         [1.0, -0.3791930672072934, 0.3353029451200118], [1.0, 0.8932538629382063, -0.647668078827123],
         [1.0, -0.6620513223156042, -0.25836299227811765], [1.0, 0.33862963691657755, 0.3237769550894922],
         [1.0, 0.6633136641274915, 0.3629747337372442], [1.0, -0.19763973216659636, -0.48223124693913055],
         [1.0, -0.7704823999144303, 0.45183352132464694], [1.0, -0.06354115415253525, 0.8502852996372192],
         [1.0, -0.9856022974424541, 0.8725836821865494], [1.0, -0.09576165651052704, -0.3683732615273472],
         [1.0, -0.5125899512340542, -0.8757013461989496], [1.0, 0.38134544154297223, -0.8818963092719321],
         [1.0, 0.7086824052562204, 0.6830835848758869], [1.0, 0.13315088112524887, -0.6292450388450543],
         [1.0, -0.1558829765440355, -0.6603156260047076], [1.0, 0.5902172105580148, -0.00975097839530914],
         [1.0, 0.2729649840259034, -0.22370314168756544], [1.0, 0.8528495793970898, 0.6025919296323612],
         [1.0, -0.10673042624403095, -0.5168125752001393], [1.0, -0.5354549927668204, -0.12217710384474811],
         [1.0, -0.036760953455707845, -0.9869093568747416], [1.0, 0.2249730618949506, 0.2090744136989724],
         [1.0, 0.7341593515336029, 0.7464401732703498], [1.0, -0.9077421498470293, 0.3007446517783263],
         [1.0, -0.7483072256954149, -0.9636538306150515], [1.0, 0.5169713906825708, -0.35694713104511555],
         [1.0, 0.48249917306626955, -0.8269842765475806], [1.0, 0.8250412258717716, 0.40647911298816175],
         [1.0, -0.018754881864012418, 0.31212254559257224], [1.0, 0.09487407678569015, -0.11166658769521232],
         [1.0, 0.047444642922708, 0.6971203378996591], [1.0, -0.7823834676923997, 0.5807753435584435],
         [1.0, -0.05729844764368086, -0.40735069339927854], [1.0, -0.9032287079515551, -0.6624189495909107],
         [1.0, 0.9423364824626319, -0.049175313379557384], [1.0, 0.9647977468810771, -0.531267205768172],
         [1.0, 0.402699801346182, -0.40991398338496676], [1.0, -0.8799764867181765, -0.8041782109143452],
         [1.0, 0.7722924480274105, -0.24789967722250927], [1.0, -0.6822848264655761, 0.699368519081992],
         [1.0, -0.4877660748288275, -0.055901903194209845], [1.0, 0.5211094409435759, -0.5433302724669256],
         [1.0, -0.9248992887196437, -0.7491497191875798], [1.0, -0.7895815720155823, -0.04559069246391223],
         [1.0, -0.595182875439767, 0.9607677138802406], [1.0, -0.7952023998495374, -0.25445670068693227],
         [1.0, -0.9344671290552455, 0.041490929192283144], [1.0, -0.7620245528908338, 0.7050524688326634],
         [1.0, 0.9927376053902175, 0.8204126990183198], [1.0, 0.02343579340628721, -0.937132084420526],
         [1.0, -0.001914822914457881, 0.883523605322079], [1.0, -0.14481325147708946, -0.7966602537748897],
         [1.0, 0.9234665259059684, 0.7673680041998048], [1.0, 0.5206837707563114, 0.7968767180891596],
         [1.0, 0.25201085976842097, 0.37395136629521475], [1.0, 0.002686067859464236, 0.3612530814140158],
         [1.0, -0.6315075455176276, -0.5986606745120981], [1.0, 0.9285403255212465, -0.19150255687090478],
         [1.0, -0.9184756195303179, -0.27123664108642265], [1.0, 0.17298835766790455, 0.2787775825786558],
         [1.0, 0.3604254823969242, -0.8915219169963482], [1.0, 0.10117035207860314, 0.03028499433067311],
         [1.0, 0.802383139188277, -0.42956038084111503], [1.0, 0.7062633163480339, 0.06455891912071654],
         [1.0, -0.168394526798733, -0.23245599617957224], [1.0, 0.4635251374002647, 0.04560522308763004],
         [1.0, 0.583236456570648, 0.36967666455507864], [1.0, 0.5659230103033279, -0.09966900631196896],
         [1.0, 0.38210371807160715, -0.38547078626631537], [1.0, -0.023624703928075252, 0.4000935635105234],
         [1.0, 0.5673968919986552, -0.9529315646651006], [1.0, 0.43596254844424887, -0.319594250068864],
         [1.0, 0.7625568386804693, -0.07788628767102668], [1.0, -0.3244000502209592, 0.24770132289902946],
         [1.0, -0.47080184519355783, 0.35873360413031663], [1.0, 0.8118792467264258, 0.9248630436418743],
         [1.0, 0.5263185515366335, 0.6416322070071163], [1.0, -0.03976921964481961, 0.4814142055378632],
         [1.0, -0.22463685642863207, -0.8460290563928596], [1.0, 0.5959216272035261, 0.22966623822591847],
         [1.0, 0.8351671846382693, 0.5929966942935772], [1.0, 0.23734673960792763, 0.7937330737855794],
         [1.0, -0.14212214628737074, -0.8894035476766042], [1.0, 0.010901494439543136, 0.019283516935186373],
         [1.0, 0.04939013889064081, -0.33537592461953536], [1.0, -0.012705910414339883, 0.2773807367817256],
         [1.0, 0.763689754564759, 0.6134709526646664], [1.0, 0.9209584824716848, -0.8656625485463592],
         [1.0, 0.6339425476807863, 0.9122597761837961], [1.0, 0.5365915125052496, 0.08176370088615315],
         [1.0, 0.9665888298600924, -0.8515356953479334], [1.0, 0.768495638493359, -0.3801264342570905],
         [1.0, 0.6960792053504254, -0.006176234425742333], [1.0, 0.9692226354978515, 0.9725813412874587],
         [1.0, 0.22639385751360042, -0.713340616512268], [1.0, 0.9331634096998407, 0.5941029900393016]]
    y = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0,
         0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    weights_old = [0.11911449718891864, 0.4362866028341834, 0.4470447225513017]


    # point a
    # run with eta = 1 and plot the linear separator
    [weights_a, epoch_a] = perceptronLearingAlgo(x, y)  # run the algorithm
    plotPlane(x, y, weights_a, weight_old=weights_old)  # plot

    # point b
    # plot errors per epoch with different eta's
    [weights_b1, epoch_b1] = perceptronLearingAlgo(x, y, eta=1)  # run the algorithm
    [weights_b2, epoch_b2] = perceptronLearingAlgo(x, y, eta=0.1)  # run the algorithm
    [weights_b3, epoch_b3] = perceptronLearingAlgo(x, y, eta=10.0)  # run the algorithm

    plotEpoch(epoch_b1, 1)  # eta 1
    plotEpoch(epoch_b2, 0.1)  # eta 1
    plotEpoch(epoch_b3, 10)  # eta 1


    # point c
    # set weight vector as Q1(a) and sample n = 1000 data points
    n = 1000
    weights_c = weights_old.copy()
    y_c = []
    x_c = []  # vector of pairs (x0, x1, x2)
    boundaries_x = [1, 1, -1, 1, -1, 1]  # boundaries for x0, x1, x2, respectively
    # generate n samples with random x vector
    for i in range(n):
        x_c.append(vectorRandomGen(3, boundaries_x))  # vector [x0, x1, x2] generation

    # fill y given x, W
    for i in range(n):
        y_c.append(stepFunct(VECT_prod(weights_c, MAT_transp(x_c[i]))))

    plotPlane(x_c, y_c, weights_old)

    [weights_algo_c, epoch_algo_c] = perceptronLearingAlgo(x_c, y_c)  # run the algorithm with eta = 1

    plotPlane(x_c, y_c, weights_algo_c, weight_old=weights_old)  # plot

    print("weight vector * point c")
    print([f"{weight:.2g}" for weight in weights_c])
    
    print("weight vector from the perceptron learning algorithm point c")
    print([f"{weight:.2g}" for weight in weights_algo_c])

    # point d
    dict_d1 = {}  # dict epoch -> errors per epoch eta = 1
    dict_d2 = {}  # dict epoch -> errors per epoch eta = 0.1
    dict_d3 = {}  # dict epoch -> errors per epoch eta = 10

    for i in range(100):
        # plot errors per epoch with different eta's
        [wd1, ed1] = perceptronLearingAlgo(x, y, eta=1, initWrandom=True)  # run the algorithm
        [wd2, ed2] = perceptronLearingAlgo(x, y, eta=0.1, initWrandom=True)  # run the algorithm
        [wd3, ed3] = perceptronLearingAlgo(x, y, eta=10.0, initWrandom=True)  # run the algorithm

        # eta = 1
        # check if we have to add more epochs
        if len(ed1) > len(dict_d1.keys()):
            # add keys
            lenD = len(dict_d1.keys())
            for j in range(len(ed1) - lenD):
                dict_d1['EP' + str(j + lenD)] = []

        # add values
        for k in range(len(ed1)):
            dict_d1['EP' + str(k)].append(ed1[k])

        # eta = 0.1
        # check if we have to add more epochs
        if len(ed2) > len(dict_d2.keys()):
            # add keys
            lenD = len(dict_d2.keys())
            for j in range(len(ed2) - lenD):
                dict_d2['EP' + str(j + lenD)] = []

        # add values
        for k in range(len(ed2)):
            dict_d2['EP' + str(k)].append(ed2[k])

        # eta = 10
        # check if we have to add more epochs
        if len(ed3) > len(dict_d3.keys()):
            # add keys
            lenD = len(dict_d3.keys())
            for j in range(len(ed3) - lenD):
                dict_d3['EP' + str(j + lenD)] = []

        # add values
        for k in range(len(ed3)):
            dict_d3['EP' + str(k)].append(ed3[k])

    # computing AVG and 10th/90th percentile
    dict_avg_d1 = AVG_epoch(dict_d1)
    dict_avg_d2 = AVG_epoch(dict_d2)
    dict_avg_d3 = AVG_epoch(dict_d3)

    [dict_perc10_d1, dict_perc90_d1] = PERC_10_90_epoch(dict_d1)
    [dict_perc10_d2, dict_perc90_d2] = PERC_10_90_epoch(dict_d2)
    [dict_perc10_d3, dict_perc90_d3] = PERC_10_90_epoch(dict_d3)

    plotAVG(dict_avg_d1, dict_perc10_d1, dict_perc90_d1,1)
    plotAVG(dict_avg_d2, dict_perc10_d2, dict_perc90_d2, 0.1)
    plotAVG(dict_avg_d3, dict_perc10_d3, dict_perc90_d3, 10)


main()