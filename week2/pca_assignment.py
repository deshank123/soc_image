def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def data2():
    x = [np.random.rand() for i in range(1000)]
    y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]



import matplotlib.pyplot as plt
import numpy as np

def std_data(nparray):
    data = np.array(nparray)
    
    # Calculate mean and standard deviation
    mean = np.mean(data, axis=1, keepdims=True)
    std_dev = np.std(data, axis=1, keepdims=True)
    
    # Standardize the data
    std_data = (data - mean) / std_dev
    
    # Return standardized data as a list
    return std_data.tolist()


def DimReduction(arr):
    data_set = np.array(arr)
    std_data_set = std_data(data_set)
    std_data_set=np.array(std_data_set)
    
    # Compute covariance matrix
    cov_matrix = np.cov(std_data_set)
    
    # Perform eigendecomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvectors based on eigenvalues (descending order)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select principal component (first eigenvector with highest eigenvalue)
    pc = eigenvectors[:, 0]
    
    # Compute slope (m) and intercept (c) of the best fit line (principal component)
    m = pc[1] / pc[0]  # Slope
    c = np.mean(std_data_set[1] -m* std_data_set[0])  # Intercept
    
    # Displaying the result using matplotlib
    plt.scatter(std_data_set[0], std_data_set[1], color="red", alpha=0.4)
    plt.plot(std_data_set[0], m * std_data_set[0] + c)
    plt.title("Best Fit Line (Principal Component)")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.grid(True)
    plt.show()
    
    print("Slope =", m)
    print("Intercept=",c)

# Apply DimReduction function to data1 and data2
DimReduction(data1())
DimReduction(data2())
