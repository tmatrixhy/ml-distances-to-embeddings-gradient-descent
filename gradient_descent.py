# tmatrixhy

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def objective_derivative(X,i,D):
    """
    input: X -> nx2 Matrix of List[List[int]] representing random x,y location
           i -> row to calculate gradient for
           D -> nxn Matrix of List[List[int]] representing relative distances

    return: X -> nx2 Matrix of List[List[int]]
    """
    summation = np.zeros([1, 2])
    for j in range(len(cities)):
        if i!=j:
            distance = np.linalg.norm(np.array(X[i]) - np.array(X[j]))
            summation+= ((distance - D[i][j]) / distance) * (np.array(X[i]) - np.array(X[j]))
    
    return summation

def gradient(X, D, alpha):
    """
    input: X -> nx2 Matrix of List[List[int]] representing random x,y location
           D -> nxn Matrix of List[List[int]] representing relative distances
           alpha -> step size parameter for gradient descent

    return: X -> nx2 Matrix of List[List[int]] representing final locations
    """
    grad = np.zeros((len(cities),2))
    grad.astype(float)
    for i in range(len(cities)):
        #for each city calculate the gradient and take a step in that direction
        grad[i] = (alpha * objective_derivative(X,i,D))
    X -= grad

    return X

def plot_map(X):
    """
    input: X -> nx2 Matrix of List[List[int]] representing x,y location

    output: png image o
    """
    x,y = X.T #splits x,y (lat/long) component for plot
    colors = "rgbymcgrk"
    col_idx = 0

    fig, ax = plt.subplots()

    for x_i, y_i, city in zip(x,y,cities):
        if col_idx % 2 == 0:
            ax.scatter(x_i,y_i,c=colors[col_idx],marker='o',label=city)
        else:
            ax.scatter(x_i,y_i,c=colors[col_idx],marker='x',label=city)
        col_idx+=1

    fig.suptitle('USA Cities')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("geographic_embedding.png", bbox_inches="tight")
    print("Image saved as geographic_embedding.png")

def main():
    """
    usage: python gradient_descent [slack] [alpha] [epochs]
        -- slack - additional slack for random initilization
        -- alpha - gradient descent step size
        -- epohs - train length
    
    output: Geographic embedding as a matplotlib image.

    use:
    >>>> python gradient_descent.py 100 .02 2000
    Image saved as geographic_embedding.png
    """
    if len(sys.argv) < 4:
        print("Usage: python gradient_descent.py [slack] [alpha] [epochs])")
        print("      -- slack - additional slack for random initilization ")
        print("      -- alpha - gradient descent step size")
        print("      -- epohs - train length")
        sys.exit()

    from given_embedding import matrix as D
    from given_embedding import cities

    slack = int(sys.argv[1])
    alpha = float(sys.argv[2])
    epoch = int(sys.argv[3])

    # convert to pandas dataframe
    D = pd.DataFrame(D)

    # set range for random initilization of city location
    x_y_range = int(D.max().max()) + slack

    # populate random locations for each cities 
    X = [[np.random.randint(x_y_range),np.random.randint(x_y_range)] 
            for n in cities]

    # calculate x,y geographic locations for cities
    for i in range(epoch):
        X = gradient(X, D, alpha)

    # plot and output map
    plot_map(X)

if __name__ == '__main__':
    main()