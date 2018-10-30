from ex1 import computeCost, gradientDescent
import numpy as np
import pandas as pd

def featureNormalize(X):
    X_norm = X.astype('float64')
    X_norm -= np.mean(X_norm)
    X_norm /= np.std(X_norm)
    return X_norm

def normalEq(X, y):
    theta = np.dot(X.transpose(), y)
    theta = np.dot(np.linalg.inv(np.dot(X.transpose(),X)),theta)
    return theta


def  main():
    data = pd.read_csv('ex1data2.txt', names=['Size','NoOfBr','Price'])
    X = data.values[:,:2]
    y = data.values[:, 2].reshape(len(X),1)
    m = len(X)
    print(f'Training samples : \n{X[:10,:]}')
    print(f'\nTarget samples : \n{y[:10,:]}\n')

    print('Normalizing Features ...\n')
    X_norm = featureNormalize(X)

    print('Augmenting column of ones to X ...\n')
    X_norm = np.append(np.ones((len(X),1)), X_norm, axis=1)
    print(f'Augmented and normalized X norm : \n{X_norm[:10, :]}')

    print('Running Gradient Descent ...\n')

    alpha = 0.01
    num_iters = 400

    theta = np.zeros((3,1))
    (theta, J_history) = gradientDescent(X_norm, y, theta, alpha, num_iters)

    print('Theta computed from gradient descent: \n');
    print(f'{theta}');
    print('\n');

    price = np.dot(np.array([[1 ,1650, 3]]),theta)
    print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {price}')

    print('Using normal equation ...\n')

    X_aug = np.append(np.ones((len(X),1)), X, axis=1)
    theta = normalEq(X_aug, y)
    print(f'theta using normal equation : \n {theta}\n')

    price = np.dot(np.array([[1, 1650, 3]]), theta)

    print(f'Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n  {price}') 


if __name__ == "__main__":
    main()
