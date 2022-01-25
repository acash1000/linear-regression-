import csv
import math
import random

import numpy as np
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a sting representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data pofloats and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            arrow = []
            # "BODYFAT","DENSITY","AGE","WEIGHT","HEIGHT","ADIPOSITY",
            # "NECK","CHEST","ABDOMEN","HIP","THIGH","KNEE","ANKLE","BICEPS","FOREARM","WRIST"
            arrow.append(float(row['BODYFAT']))
            arrow.append(float(row['DENSITY']))
            arrow.append(float(row['AGE']))
            arrow.append(float(row['WEIGHT']))
            arrow.append(float(row['HEIGHT']))
            arrow.append(float(row['ADIPOSITY']))
            arrow.append(float(row['NECK']))
            arrow.append(float(row['CHEST']))
            arrow.append(float(row['ABDOMEN']))
            arrow.append(float(row['HIP']))
            arrow.append(float(row['THIGH']))
            arrow.append(float(row['KNEE']))
            arrow.append(float(row['ANKLE']))
            arrow.append(float(row['BICEPS']))
            arrow.append(float(row['FOREARM']))
            arrow.append(float(row['WRIST']))
            dataset.append(arrow)
    dataset = np.array(dataset)
    return dataset


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    count = 0
    summation = 0
    for row in dataset:
        summation += row[col]
        count += 1
    xbar = (1 / count) * summation
    summation2 = 0
    for row in dataset:
        summation2 += (row[col] - xbar) ** 2
    outside = (1 / (count - 1))
    inside = outside * summation2
    standereddev = math.sqrt(inside)
    print(count)
    print(round(xbar, 2))
    print(round(standereddev, 2))


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    beta0 = betas[0]
    summation = 0
    beta = betas[1:]
    n = 0
    for i in range(len(dataset)):
        temp = np.zeros((1, len(cols)))
        counter = 0
        for j in cols:
            temp[0][counter] = dataset[i][j]
            counter += 1
        summation += (np.dot(temp, beta) + beta0 - dataset[i][0]) ** 2
        n += 1

    mse = float(summation / n)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    beta0 = betas[0]
    beta = betas[1:]
    n = 0
    resarry = np.zeros((1, len(betas)))
    for i in range(len(dataset)):
        temp = np.zeros((1, len(cols)))
        counter = 0
        for j in cols:
            temp[0][counter] = dataset[i][j]
            counter += 1
        dotpract =np.dot(temp, beta)
        summation  = ((dotpract + beta0) - dataset[i][0])
        resarry[0][0] += summation
        for j in range(len(temp[0])):
            resarry[0][j+1] += summation*temp[0][j]
        n += 1
    resarry = resarry*(2/n)
    grads = resarry
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    beta = gradient_descent(dataset, cols, betas)
    for bet in range(len(betas)):
        sec = (eta * beta[0][bet])
        temps =betas[bet] - sec
        beta[0][bet] = round(float(temps),2)
    reg = regression(dataset, cols, list(beta[0]))
    t = 1
    while (t <= T):
        st = []
        st.append( str(t))
        st.append(str(reg))
        for bet in beta[0]:
            st.append( str(round(bet,2)))
            # if (not (bet == beta[0][len(beta[0]) - 1])):
            #    st.append(" ")
        bettemp = beta.copy()
        beta = gradient_descent(dataset, cols, list(beta[0]))
        for bet in range(len(betas)):
            sec = (eta * beta[0][bet])
            temps = bettemp[0][bet] - sec
            beta[0][bet] = round (float(temps),2)
        reg = regression(dataset, cols, list(beta[0]))
        print('{st[0]} {st[1]} {st[2]} {st[3]}'.format(st=st))
        t+=1


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = []
    lables = dataset[:,0]
    x= dataset[:,cols]
    ones = np.ones((len(dataset),1))
    x= np.hstack((ones,x))
    # x = np.reshape(x,(len(x),1))
    xt = np.transpose(x)
    xtx = xt@x
    xxtinverse = np.linalg.inv(xtx)
    xty = xt.dot(lables)
    beta = xxtinverse.dot(xty)
    for i in range(len(beta)):
        betas.append(beta[i])
    mse = regression(dataset,cols,betas)
    return (mse,*betas)

def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    tup =compute_betas(dataset,cols)
    result =tup[1]
    for i in range(len(features)):
        result+= tup[i+2] *features[i]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    #𝑦𝑖=𝛽0+𝛽1𝑥𝑖+𝑧𝑖
    z =np.random.normal(0,sigma)
    b0 = betas[0]
    a0 = alphas[0]
    lineardata = np.zeros((len(X[0]),2))
    quadraticdata = np.zeros((len(X[0]),2))
    for i in range(len(X[0])):
        yilin = b0+X[0][i]*betas[i+1]+z
        yiquad = a0+((X[0][i])**2)*alphas[i+1]+z
        lineardata[i][0] = yilin
        lineardata[i][1] = X[0][i]
        quadraticdata[i][0] = yiquad
        quadraticdata[i][1] = X[0][i]
    return lineardata, quadraticdata


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
    X = []
    for i in range(1000):
        X.append(random.randint(-100,100))
    X = np.array(X)
    X = X.reshape((1,1000))
    betas = [1]
    alphas = [1]
    for i in range(1000):
        beta =np.random.randint(1,2)
        alpha = np.random.randint(1,2)
        betas.append(beta)
        alphas.append(alpha)
    betas = np.array(betas)
    alphas =np.array(alphas)
    sigmas =[.0001,.001,.01,1,10,100,1000,10000,100000]
    linear = []
    quadratic = []
    for sigma in sigmas:
        linear.append(synthetic_datasets(betas,alphas,X,sigma)[0])
        quadratic.append(synthetic_datasets(betas,alphas,X,sigma)[1])
    combetasoutlinear = []
    combetasoutquad = []
    for i in range(len(linear)):
        dataset = linear[i]
        combetasoutlinear.append((compute_betas(dataset,[1])[0]))
        dataset = quadratic[i]
        combetasoutquad.append((compute_betas(quadratic[i],[1])[0]))
    plt.scatter(sigmas,combetasoutlinear)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Log(MSE)")
    plt.xlabel("Sigma")
    plt.plot(sigmas,combetasoutlinear,label="linear")
    plt.scatter(sigmas,combetasoutquad)
    plt.plot(sigmas,combetasoutquad,label="quadratic")
    plt.legend()
    plt.savefig("mse.pdf")
    # TODO: Generate datasets and plot an MSE-sigma graph

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()

data = get_dataset('bodyfat.csv')
# print(data)
# print_stats(data,1)
# print(regression(data, cols=[2, 3], betas=[0, 0, 0]))
# print(regression(data, cols=[2, 3, 4], betas=[0, -1.1, -.2, 3]))
# print(gradient_descent(data, cols=[2,3], betas=[0,0,0]))
# betas=[400,-400,300]
# foo =gradient_descent(data, cols=[1,8], betas=[400,-400,300])
# l = [394.45 ,-405.84 ,-220.18]
# for v in range(len(l)):
#     l[v] = (l[v]-betas[v])/-.0001
# print(l)
# print(foo)
#4,360.5211111099
# iterate_gradient(data, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4)
print(predict(data, cols=[1,2], features=[1.0708, 23]))
