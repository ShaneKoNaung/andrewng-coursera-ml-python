import pandas as pd
import numpy as np

def computeCost(X, y, theta):
    m = len(X)
    J = (1/(2 * m)) * np.sum((np.dot(X,theta) - y) ** 2)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(X)
    J_history = np.zeros((iterations,1))

    for i in range(iterations):

        theta = theta - (alpha/m) * np.sum((np.dot(X,theta) - y) * X, axis=0).reshape(np.shape(theta))

        J_history[i] = computeCost(X,y,theta)

    return (theta, J_history)


def main():
    # import dataset
    data1 = pd.read_csv('ex1data1.txt', names=['X','y'])

    X = data1.X
    X = X.values.reshape((len(X),1))
    y = data1.y
    y = y.values.reshape((len(y),1))

    # data shape and sample
    print('Training set :')
    print(f'{X.shape}')
    print(f'{X[:10,:]}')
    print()
    print('Test set')
    print(f'{y.shape}')
    print(f'{X[:10,:]}')
    print()

    print('adding a column of ones to X')
    X = np.append(np.ones((len(X),1)), X, axis=1)
    print('Training set now...')
    print(f'{X[:10,:]}')
    print()

    alpha = 0.01
    iterations =1500
    theta = np.zeros((2,1))

    print('Testing the cost function ...')
    J = computeCost(X, y, theta)
    print(f'With theta = [0 ; 0]\nCost computed = {J}')
    print('Expected cost value (approx) 32.07\n');


    J = computeCost(X, y, np.array([[-1],[2]]));
    print(f'\nWith theta = [-1 ; 2]\nCost computed = {J}');
    print('Expected cost value (approx) 54.24\n');

    print('\nRunning Gradient Descent ...\n')

    (theta, J_history) = gradientDescent(X, y, theta, alpha, iterations);

    print('Theta found by gradient descent:\n');
    print(f'{theta}');
    print('Expected theta values (approx)\n');
    print(' -3.6303\n  1.1664\n\n');

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot(np.array([1, 3.5]),theta);
    print(f'For population = 35,000, we predict a profit of {predict1*10000}');
    predict2 = np.dot(np.array([1, 7]) ,theta);
    print(f'For population = 70,000, we predict a profit of {predict2*10000}');


if __name__ == "__main__":
    main()
