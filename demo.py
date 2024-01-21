import numpy as np
from sklearn.cluster import KMeans

# Function to get data input dynamically
def get_data_input():
    num_points = int(input("Enter the number of data points: "))
    X = np.empty((num_points, 2))  # Create an empty NumPy array with 2 dimensions
    for i in range(num_points):
        point_x = float(input(f"Enter x-coordinate for point {i + 1}: "))
        point_y = float(input(f"Enter y-coordinate for point {i + 1}: "))
        X[i] = [point_x, point_y]  # Add the coordinates to the array
    return X

# Get input data from the user
X = get_data_input()

# Number of clusters
k = int(input("Enter the number of clusters: "))

# Create KMeans object and fit it to the data
kmeans = KMeans(n_clusters=k, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Print the results
print("Cluster labels:", y_kmeans)
print("Cluster centers:", kmeans.cluster_centers_)
