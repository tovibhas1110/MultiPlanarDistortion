import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle

with open("files200.pkl", "rb") as f:
    workspace = pickle.load(f)

globals().update(workspace)


# Load MATLAB file
# data = loadmat("files500.mat")

# Extract variables
# uSuper = data["uSuper"]
# nely = int(data["nely"].squeeze())
# subsNumlayer = int(data["subsNumlayer"].squeeze())
# nelx = int(data["nelx"].squeeze())

# xTopo = data["xTopo"]
# subsFilter = data["subsFilter"]
# edofMat_ihs = data["edofMat_ihs"]

# XNC = data["XNC"]
# YNC = data["YNC"]


# ---------------------------------------------------------
# Plotting Distortion
# ---------------------------------------------------------

# U_super_ihs_grid_x
U_super_ihs_grid_x = np.reshape(np.sum(np.squeeze(uSuper)[0::2, :],axis=1),
                 (nely + subsNumlayer + 1, nelx + 1),
    order="F")

# U_super_ihs_grid_y
U_super_ihs_grid_y = np.reshape(np.sum(np.squeeze(uSuper)[1::2, :],axis=1),
                 (nely + subsNumlayer + 1, nelx + 1),
    order="F")


# Dummy variables
dummyxPhys_ihs = (
    xTopo.ravel(order='F') *
    subsFilter.ravel(order='F')
)

# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig, ax = plt.subplots()

# MATLAB:
# for i = 1:(nely+subsNumlayer)*nelx
#
# MATLAB uses 1-based indexing, Python uses 0-based indexing
for i in range((nely + subsNumlayer) * nelx):

    if dummyxPhys_ihs[i] == 1:

        # MATLAB:
        # dummyNodeIds = [edofMat_ihs(i,2:2:end)./2, ...
        #                 edofMat_ihs(i,2)./2];

        # Convert MATLAB indices to Python indices.
        # MATLAB edofMat contains DOF numbers, where:
        # x DOF = 1, y DOF = 2
        #
        # Taking 2:2:end gives y DOFs.
        dummyNodeIds = np.concatenate([
            edofMat_ihs[i, 1::2] / 2,
            [edofMat_ihs[i, 1] / 2]
        ])


        # MATLAB node IDs are 1-based
        dummyNodeIds = dummyNodeIds.astype(int) - 1

        dummyXCoor = XNC.ravel(order='F')
        dummyYCoor = YNC.ravel(order='F')

        # Plot original geometry
        ax.plot(
            dummyXCoor[dummyNodeIds],
            dummyYCoor[dummyNodeIds],
            color="k",
            linewidth=2
        )


# ---------------------------------------------------------
# Deformed coordinates
# ---------------------------------------------------------

deformFactor = 1

XDeform = XNC + deformFactor * U_super_ihs_grid_x
YDeform = YNC + deformFactor * U_super_ihs_grid_y


# ---------------------------------------------------------
# Plot deformed geometry
# ---------------------------------------------------------

for i in range((nely + subsNumlayer) * nelx):

    if dummyxPhys_ihs[i] == 1:

        dummyNodeIds = np.concatenate([
            edofMat_ihs[i, 1::2] / 2,
            [edofMat_ihs[i, 1] / 2]
        ])

        dummyNodeIds = dummyNodeIds.astype(int) - 1

        dummyXCoor = XDeform.flatten(order="F")
        dummyYCoor = YDeform.flatten(order="F")

        ax.plot(
            dummyXCoor[dummyNodeIds],
            dummyYCoor[dummyNodeIds],
            linestyle="-",
            color="r",
            linewidth=2
        )


# ---------------------------------------------------------
# MATLAB:
# axis equal
# axis off
# axis tight
# ---------------------------------------------------------

ax.set_aspect("equal")
ax.axis("off")
ax.autoscale(enable=True, tight=True)

plt.show()