import numpy as np
from import_OE_data import import_OE_data
from import_anipose_data import import_anipose_data 
from import_KS_data import import_KS_data
import colorlover as cl
from collections import deque
from config import config as CFG
from MUsim import MUsim


# Load the two datasets into numpy arrays
# session1_data = np.load('/home/tony/git/rat-loco/20221116-3_godzilla_speed05_incline00_phase.npy')
# session2_data = np.load('/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy')
mu = MUsim()
session1_data = np.load('/home/tony/git/rat-loco/20221116-3_godzilla_speed05_incline00_phase.npy')
session2_data = np.load('/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy')

# Get the number of trajectories in each dataset
num_trajectories_session1 = session1_data.shape[0]
num_trajectories_session2 = session2_data.shape[0]

# Get length of all trajectories
len_trajectories_session1 = session1_data.shape[1]
len_trajectories_session2 = session2_data.shape[1]
assert len_trajectories_session1 == len_trajectories_session2, "Length of arrays should be the same!"
len_trajectories_session = len_trajectories_session1

# Create a matrix to store the results
Euclid_pair = np.zeros((num_trajectories_session1, num_trajectories_session2))

# Loop over all pairs of trajectories
for i in range(num_trajectories_session1//10):
    for j in range(num_trajectories_session2//10):
        # for iPoint in range(len_trajectories_session):
            # Calculate the Euclidean distance between the two trajectories
        distance = np.linalg.norm(session1_data[i][iPoint] - session2_data[j][iPoint])
            # Sum the pair-wise Euclidean distance of all points in order
            Euclid_pair = np.sum(distance)

# Print the results
print(Euclid_pair)
print(session1_data[i] - session2_data[j])
# print(session1_data[i])

# Get the number of points in each dataset
#num_points_session1 = session1_data.shape[0] * session1_data.shape[1]
#num_points_session2 = session2_data.shape[0] * session2_data.shape[1]

# Reshape the datasets into 2D arrays with shape (num_points, 3)
#session1_data_2d = session1_data.reshape((num_points_session1, 3))
#session2_data_2d = session2_data.reshape((num_points_session2, 3))

# Calculate the Euclidean distance between all pairs of points
#Euclid_all = 0.0
# for i in range(num_points_session1):
#     for j in range(num_points_session2):
#         # Calculate the Euclidean distance between the two points
#         distance = np.linalg.norm(session1_data_2d[i] - session2_data_2d[j])
#         # Add the distance to the sum
#         Euclid_all += distance

# # Print the sum of distances
# print(Euclid_all)





