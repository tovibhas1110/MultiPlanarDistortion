import numpy as np
import os
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import scipy.io  # for loading .mat files
from scipy.sparse.linalg import spsolve
from mmasub import mmasub
import pdb
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle

np.set_printoptions(
    threshold=np.inf,
    precision=6,
    suppress=True
)

# === MODEL DEFINITION ===

# Design Domain
nelx = 80
nely = 80

# Substrate Domain
subsNumlayer = 1
subs = np.ones((subsNumlayer, nelx))
subsFilter = 1 - np.vstack([np.zeros((nely, nelx)),subs])

# Total elements
totalEles = nelx * (nely + subsNumlayer)
#print("Total Elements:",totalEles)

# Total nodes
totalNodes = (nelx + 1) * (nely + subsNumlayer + 1)
#print("Total Nodes:",totalNodes)

# Penalization coefficients
penalKappa = 3
penalCp = 2
penalIHS = 3
penalIHSf = 4

# Filter radius
rminComp = 5

# === MATERIAL PROPERTIES ===

# Thermal conductivity
kappaMat = 45e-3
kappaMin = kappaMat * 1e-9

# Heat capacity
cpMat = (7800 * 496) * 1e-9
cpMin = cpMat * 1e-3

# Thermal diffusivity
diffMat = kappaMat / cpMat

# Thermal expansion coefficient
expMat = 9e-6

# Temperature difference: melting and room temperature
deltaTemp = 1500 - 20

# Young's modulus
EMat = 210000
EMin = EMat * 1e-9

# Poisson's ratio
nuMat = 0.3

# Calculation of the thermal inherent strain
ihsStrain_xx = -expMat * deltaTemp
ihsStrain_yy = -expMat * deltaTemp

ihsStrain = np.array([
    ihsStrain_xx,
    ihsStrain_yy,
    0
]).reshape(-1, 1)

#print('Strain Array:', ihsStrain)

# Strain Displacement Relationship
B = np.array([
    [-0.5,  0,   0.5, 0,    0.5, 0,    -0.5, 0],
    [0,    -0.5, 0,   -0.5,  0,  0.5,  0,   0.5],
    [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5]
])

#print('Strain-Displacement Relationship Matrix:',B)

# Stress Strain Relationship
D = ((EMat - EMin) / (1 - nuMat**2)) * np.array([
    [1,      nuMat, 0],
    [nuMat,  1,     0],
    [0,       0,    0.5 * (1 - nuMat)]
])

#print('Constitutive Law Matrix:',D)

# Load calculation from inherent strain
ihsStress = D @ ihsStrain
ihsLoad = B.T @ ihsStress

#print('Stress Calculation:',ihsStress)
#print('Equivalent Load:',ihsLoad)

# === Coordinate axis ===

# Node coordinates
XNC, YNC = np.meshgrid(np.arange(0, nelx + 1),np.arange(0, nely + subsNumlayer + 1))
YNC = np.flipud(YNC)

#print('Nodal Coordinates X:',XNC)
#print('Nodal Coordinates Y:',YNC)

# Element centroid coordinates
XEC, YEC = np.meshgrid(np.arange(1, nelx + 1),np.arange(1, nely + subsNumlayer + 1))
XEC = XEC - 0.5
YEC = np.flipud(YEC) - 0.5

#print('Element Coordinates X:',XEC)
#print('Element Coordinates Y:',YEC)

# === Process Definition ===

# Total number of processes
numPI = 2

# Total time
totalTime = 1

# Time stamps to define process intervals
PI = 1 - (np.arange(numPI, -1, -1) / numPI)
PI[-1] = PI[-1] + 0.1

#print('Process Intervals:', PI)
# Orientation definition in each process interval
thetaPhys = (np.pi / 180) * np.array([85, 45])

#print('Layer Orientations:', thetaPhys)

# Length of a bead
bead = 3

#print('Bead Size:', bead)

# === Initialization of the parameters for Differental representation of MP===
paraH1 = 100
paraH2 = 100
shapeH2 = 4

# === FE ANALYSIS - Thermal Analysis - PDE Solve ===

# Element Matrix
KE_T = (
    np.array([
        [2, -2, -1,  1],
        [-2, 2,  1, -1],
        [-1, 1,  2, -2],
        [1, -1, -2, 2]
    ]) / 6
    +
    np.array([
        [2,  1, -1, -2],
        [1,  2, -2, -1],
        [-1, -2, 2,  1],
        [-2, -1, 1,  2]
    ]) / 6
)

CE_T = np.array([
    [4, 2, 1, 2],
    [2, 4, 2, 1],
    [1, 2, 4, 2],
    [2, 1, 2, 4]
]) / 36

#print('KE thermal:', KE_T)
#print('CE thermal:', CE_T)

# Node numbering
nodenrs_T = np.arange(1, totalNodes + 1).reshape(nely + subsNumlayer + 1, nelx + 1, order='F')

# First DOFs
edofVec_T = nodenrs_T[1:, :-1].reshape(totalEles, 1, order='F')

# Connecivity
edofMat_T = (
    np.tile(edofVec_T, (1, 4))
    +
    np.tile(np.array([0, (nely + subsNumlayer) + 1, (nely + subsNumlayer), -1]),(totalEles, 1))
)

# i and j index of the K matrix
iK_T = np.kron(
    edofMat_T,
    np.ones((4, 1))
).T.reshape(16 * totalEles, order='F')

jK_T = np.kron(
    edofMat_T,
    np.ones((1, 4))
).T.reshape(16 * totalEles, order='F')

# Element numbering
elenrs_T = np.arange(1,totalEles+1).reshape(nely+subsNumlayer,nelx, order='F')

#print('Node Numbers:',nodenrs_T)
#print('First DOF:',edofVec_T)
#print('Connectivity:', edofMat_T)
#print('Ki Index:',iK_T)
#print('Kj Index:',jK_T)
#print('Element Numbering:', elenrs_T)

# Initialize the arrays
Q_T = lil_matrix((totalNodes, 1))
TN_T = lil_matrix((totalNodes, 1))

# Identifying dofs to apply boundary condition
ConstTStartdof_T = nodenrs_T[nely+subsNumlayer,:].reshape(-1, 1)

# Applying Boundary condition
TN_T[ConstTStartdof_T - 1, 0] = 1

# Arrays initialized with previous and next step
T0 = TN_T
Tp = TN_T
Tn = TN_T

# Total dofs
alldofs_T = np.arange(1,totalNodes+1).reshape(-1,1)

# Known DOFs
knowndofs_T = np.sort(ConstTStartdof_T.flatten()).reshape(-1,1)

# Unknown DOFs
unknowndofs_T = np.setdiff1d(alldofs_T, knowndofs_T).reshape(-1,1)

#print('Q_T Array:',Q_T.shape)
#print('TN_T Array:',TN_T)
#print('Constraint DOFs',ConstTStartdof_T)
#print('T0:',T0)
#print('Tp:',Tp)
#print('Tn:',Tn)
#print('All Dofs:',alldofs_T)
#print('Known Dofs:',knowndofs_T)
#print('Unknown Dofs:',unknowndofs_T)


# === FE Analysis Inherent Strain ===

# Element Matrix
A11_ihs = np.array([[12,  3, -6, -3],
                    [3, 12,  3,  0],
                    [-6,  3, 12, -3],
                    [-3,  0, -3, 12]])

A12_ihs = np.array([[-6, -3,  0,  3],
                    [-3, -6, -3, -6],
                    [0, -3, -6,  3],
                    [3, -6,  3, -6]])

B11_ihs = np.array([[-4,  3, -2,  9],
                    [3, -4, -9,  4],
                    [-2, -9, -4, -3],
                    [9,  4, -3, -4]])

B12_ihs = np.array([ [2, -3,  4, -9], 
                    [-3,  2,  9, -2],
                    [4,  9,  2,  3], 
                    [-9, -2,  3,  2]])

KE_ihs = (
    1 / (1 - nuMat**2) / 24
    * (
        np.block([
            [A11_ihs, A12_ihs],
            [A12_ihs.T, A11_ihs]
        ])
        + nuMat
        * np.block([
            [B11_ihs, B12_ihs],
            [B12_ihs.T, B11_ihs]
        ])
    )
)

#print('KE_ihs:',KE_ihs)

# Connectivity Definitions

# Node numbering
nodenrs_ihs = np.arange(1,totalNodes+1).reshape(1+nely+subsNumlayer, 1+nelx, order='F')

# First DOFs
edofVec_ihs = (2*nodenrs_ihs[:-1,:-1] + 1).reshape(-1,1,order='F')

# Connectivity
edofMat_ihs = (np.tile(edofVec_ihs,(1,8))
            + np.tile(np.array([0, 1, 2*(nely+subsNumlayer)+2, 2*(nely+subsNumlayer)+3,2*(nely+subsNumlayer)+0,2*(nely+subsNumlayer)+1 ,-2, -1]),(totalEles,1))
)

# i and j index of the K matrix
iK_ihs = np.kron(
    edofMat_ihs,
    np.ones((8, 1))
).T.reshape(64 * totalEles, order='F')

jK_ihs = np.kron(
    edofMat_ihs,
    np.ones((1, 8))
).T.reshape(64 * totalEles, order='F')

# Initializing the array
U_ihs = np.zeros((2*totalNodes,1))
U_ihs_fd = np.zeros((2*totalNodes,1))
lambda_ihs = np.zeros((2*totalNodes,1))
U_super_ihs = np.zeros((2*totalNodes,1))

# Defining fixed DOFs
fixeddofs_ihs1 = 2 * nodenrs_ihs[-2:, :] - 1
fixeddofs_ihs2 = 2 * nodenrs_ihs[-2:, :]
fixeddofs_ihs = np.vstack((
    fixeddofs_ihs1.ravel(),
    fixeddofs_ihs2.ravel()
))

# Total DOFs
alldofs_ihs = np.arange(1, 2 * totalNodes + 1)

# Unknown/free DOFs
freedofs_ihs = np.setdiff1d(alldofs_ihs, fixeddofs_ihs.ravel())

#print('Fixed Dofs:',fixeddofs_ihs)
#print('Free Dofs:', freedofs_ihs)

# === Density Filter ===

# Size of the filter based on rminComp
filter_radius = int(np.ceil(rminComp)) - 1
nely_total = nely + subsNumlayer
max_entries = totalEles * (2 * filter_radius + 1) ** 2

iH_C = np.ones(max_entries, dtype=int)
jH_C = np.ones(max_entries, dtype=int)
sH_C = np.zeros(max_entries)

k_C = 0

for i1_C in range(1, nelx + 1):
    for j1_C in range(1, nely_total + 1):

        e1_C = (i1_C - 1) * nely_total + j1_C

        for i2_C in range(max(i1_C - filter_radius, 1), min(i1_C + filter_radius, nelx) + 1):
            for j2_C in range(max(j1_C - filter_radius, 1), min(j1_C + filter_radius, nely_total) + 1):

                e2_C = (i2_C - 1) * nely_total + j2_C

                iH_C[k_C] = e1_C
                jH_C[k_C] = e2_C
                sH_C[k_C] = max(0,rminComp- np.sqrt((i1_C - i2_C) ** 2 + (j1_C - j2_C) ** 2))

                k_C += 1

# MATLAB sparse uses 1-based indices, SciPy uses 0-based indices
H_C = coo_matrix((sH_C, (iH_C - 1, jH_C - 1)),shape=(totalEles, totalEles)).tocsr()

# Sum of weights for each element
Hs_C = np.asarray(H_C.sum(axis=1))


# === Definition of the geometry ===

# Design Variable
x = np.ones((nely, nelx))

# Defining the design using curve variable
# Simple wall
curve = x.copy()

# Wang Tapered Wall
curve = np.block([
    [
        np.zeros((nely // 2, nelx // 2)),
        np.ones((nely // 2, nelx // 2))
    ],
    [
        np.ones((nely // 2, nelx // 2)),
        np.fliplr(
            np.triu(np.ones((nely // 2, nelx // 2)))
        )
    ]
])

# Integrating design in design variable
curveIndex = np.where(curve.ravel() <= 0)[0]

xTopo = x.ravel().copy()

xTopo[curveIndex] = 0

xTopo = xTopo.reshape(nely, nelx)

xTopo = np.vstack((xTopo, subs))

# Design variable for inherent strain
xPhys_ihs = np.vstack((x, subs))

# Design variable for thermal calculation
xPhys_T = np.vstack((x, subs))

# === Interested Elements and Nodes ===
LE = np.ones((totalEles,1))
LN = np.ones((2*totalNodes,1))

# === Optimization loop starts ===

# loop Definition
loop = 0
maxloop = 500

# Number of constraints and variables for MMA
m = numPI
n = totalEles + numPI

# MMA parameters
a0mma = 1
amma = np.zeros(m)[:,None]
cmma = 10000 * np.ones(m)[:,None]
dmma = np.zeros(m)[:,None]

# Initialize old values of design variables
xold1 = np.concatenate((xPhys_T.ravel(order='F'),thetaPhys.ravel(order='F')))[:,None]
xold2 = np.concatenate((xPhys_T.ravel(order='F'),thetaPhys.ravel(order='F')))[:,None]

# Initialize low and upp
low = np.concatenate((np.zeros(totalEles),np.zeros(numPI)))[:,None]
upp = np.concatenate((np.ones(totalEles),90.0 * (np.pi / 180) * np.ones(numPI)))[:,None]

# Move limits
movekappa = 0.2
movetheta = 0.5 * np.pi / 180

# Initialization of the trackers
objTrack = np.empty((0, 11))

# Boundary Definition
consBoundEps = 0.1

while loop < maxloop:

    loop += 1

    # Reinitialization of temperature
    Tp = T0.copy()
    Tn = T0.copy()

    factor = np.sum(xTopo.ravel(order='F') * xPhys_T.ravel(order='F')) / np.sum(xTopo.ravel(order='F'))
    dt = 5 * (np.sqrt(nelx**2 + nely**2)/ (factor * diffMat))
    alphaEuler = 1


    # Thermal conductivity matrix
    kappa = (kappaMin + (xTopo.ravel(order='F') * xPhys_T.ravel(order='F'))**penalKappa* (kappaMat - kappaMin))
    sK_T = (KE_T.ravel(order='F')[:, None] * kappa[None, :]).ravel(order='F')
    K_T = coo_matrix((sK_T, (iK_T.ravel() - 1, jK_T.ravel() - 1)),shape=(totalNodes, totalNodes)).tocsr()
    K_T = (K_T + K_T.T) / 2


    # Specific heat capacity matrix
    cp = (cpMin + (xTopo.ravel(order='F') * xPhys_T.ravel(order='F'))**penalCp* (cpMat - cpMin))
    sCp_T = (CE_T.ravel(order='F')[:, None] * cp[None, :]).ravel(order='F')
    Cp_T = coo_matrix((sCp_T, (iK_T.ravel() - 1, jK_T.ravel() - 1)),shape=(totalNodes, totalNodes)).tocsr()
    Cp_T = (Cp_T + Cp_T.T) / 2

    # Defining the solving matrices
    Et_T = (Cp_T / dt) + (alphaEuler * K_T)
    At_T = (-(Cp_T / dt)) + ((1 - alphaEuler) * K_T)

    # MATLAB DOFs are 1-based; Python DOFs are 0-based
    unknown = unknowndofs_T.ravel() - 1
    known = knowndofs_T.ravel() - 1
    

    # Solve the heat equation
    rhs = (
        Q_T[unknown]
        - At_T[unknown, :] @ Tp
        - Et_T[unknown, :][:, known] @ Tn[known]
    )

    Tn[unknown] = spsolve(
        Et_T[unknown, :][:, unknown],
        rhs
    )

    # Temperature at the current time step
    TN_T = (1 - alphaEuler) * Tp + alphaEuler * Tn
    TN_T_array = Tn.toarray()
    TP_T_array = Tp.toarray()

    # TN_T_node = np.asarray(TN_T).reshape((nely + subsNumlayer + 1, nelx + 1),order='F')
    TN_T_node = TN_T.toarray().reshape((nely + subsNumlayer + 1, nelx + 1),order='F')

    # Element temperature
    TE_T = np.mean(TN_T_array[edofMat_T.T-1],axis=0)
    TE_T = TE_T.ravel().reshape((nely + subsNumlayer, nelx),order='F')

    # === Plotting Element Temperature ===
    """
    plt.figure(1)
    plt.clf()
    plt.imshow(
        TE_T,
        vmin=0,
        vmax=1,
        aspect='equal'
        )
    plt.axis('off')
    plt.colorbar()
    plt.show()
    """

    # === Defining the time field ====
    tPhys = 1 - TE_T

    # === Plotting fictitious time field ===
    """
    plt.figure(2)
    plt.clf()
    # tPhys_visual = xTopo .* tPhys
    tPhys_visual = xTopo * tPhys
    tPhys_visual = tPhys_visual.ravel(order='F').copy()
    # Set locations where xTopo == 0 to NaN
    xTopo_flat = xTopo.ravel(order='F')
    tPhys_visual[xTopo_flat == 0] = np.nan
    # Reshape back to element grid
    tPhys_visual = tPhys_visual.reshape((nely + subsNumlayer, nelx),order='F')
    # Plot
    plt.imshow(tPhys_visual[:nely, :nelx],vmin=0,vmax=1,aspect='equal',origin='upper')
    # Contour
    plt.contour(XEC[:nely, :nelx],np.flipud(YEC[:nely, :nelx]),tPhys_visual[:nely, :nelx],levels=PI,colors='k')
    # Add contour labels
    plt.clabel(plt.gca().collections[-1],inline=True,fontsize=8)
    # Parula-like colormap
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('white')
    plt.set_cmap(cmap)
    # Colorbar
    plt.colorbar()
    plt.axis('off')
    plt.axis('equal')
    plt.show()

    plt.figure(3)
    plt.clf()
    tPhys_visual = xTopo * xPhys_T
    tPhys_visual = tPhys_visual.ravel(order='F').copy()
    # Set locations where xTopo == 0 to NaN
    xTopo_flat = xTopo.ravel(order='F')
    tPhys_visual[xTopo_flat == 0] = np.nan
    # Reshape back to element grid
    tPhys_visual = tPhys_visual.reshape((nely + subsNumlayer, nelx),order='F')
    # Plot
    plt.imshow(tPhys_visual[:nely, :nelx],vmin=0,vmax=1,aspect='equal',origin='upper')
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('white')
    plt.set_cmap(cmap)
    # Colorbar
    plt.colorbar()
    plt.axis('off')
    plt.axis('equal')
    plt.show()
    """

    # === Segmenting the design into process intervals ===

    # Total maximum layer in the design domain
    maxLayer = int(np.ceil((1 / bead) * np.sqrt((np.max(XNC.ravel()) - np.min(XNC.ravel()))**2+ (np.max(YNC.ravel()) - np.min(YNC.ravel()))**2)))

    # Initialize array to identify elements below the time stamp
    eBelowPI = np.zeros((totalEles, 1, numPI + 1))
    
    for i in range(1, numPI + 1):
        eBelowPI[:, 0, i] = (1- 1 / (1+ np.exp(-paraH1* (tPhys.ravel(order='F')- PI[i]))))

    eInPI = (xTopo.ravel(order='F')[:, None]* (eBelowPI[:, 0, 1:]- eBelowPI[:, 0, :-1]))
    totalEInPI = np.sum(eInPI, axis=0)

    # Orientation Field
    phi = (thetaPhys.ravel(order='F')[:, None]).T * (1 + 0 * eInPI)
    distPhi = (np.sin(phi) * YEC.ravel(order='F')[:, None] + np.cos(phi) * XEC.ravel(order='F')[:, None])
    
    # New layer calculation
    newLayer = np.zeros((totalEles, maxLayer, numPI))

    for i in range(numPI):
        for j in range(maxLayer):
            newLayer[:, j, i] = (eInPI[:, i]* (1- 1 / (1+ np.exp(paraH2* (1- (shapeH2 / bead**2)* (distPhi[:, i]- bead / 2- bead * j)**2)))))

    depoStruct = np.cumsum(newLayer, axis=1)

    for i in range(1, numPI):
        depoStruct[:, :, i] += depoStruct[:, -1, i - 1][:, None]

    subFilterIndexMat = np.where(subsFilter.ravel(order='F') == 0)[0]
    depoStruct[subFilterIndexMat, :, :] = 1

    # === Force calculation ===
    fLayerInPI = np.zeros((2 * totalNodes, maxLayer, numPI))

    for i in range(numPI):
        for j in range(maxLayer):

            dummy_newLayer = (xTopo.ravel(order='F')* subsFilter.ravel(order='F')* newLayer[:, j, i])
            
            if penalIHSf != 0:
                dummy_newLayer_kron = np.kron(dummy_newLayer ** penalIHSf,np.ones(ihsLoad.shape[0]))
            else:
                dummy_newLayer_kron = np.kron(1 + 0 * dummy_newLayer,np.ones(ihsLoad.shape[0]))

            dummy_ihsload = np.tile(ihsLoad, (totalEles, 1)).ravel()
            dummy_edofMat = edofMat_ihs.ravel()
            
            # MATLAB sparse() uses 1-based indices; SciPy uses 0-based
            dummy_fLayerInPI = coo_matrix(( dummy_newLayer_kron * dummy_ihsload.ravel(order='F'),
                    (dummy_edofMat - 1, np.zeros(dummy_edofMat.size, dtype=int))),
                shape=(2 * totalNodes, 1)).toarray().ravel()
            
            fLayerInPI[:, j, i] = dummy_fLayerInPI

    # === Finite Element - Displacement Calculation ===
    uLayerInPI = np.zeros((2 * totalNodes, maxLayer, numPI))
    dummy_uLayerInPI = np.zeros(2 * totalNodes)
    fAfterDepoInPI = np.zeros((2 * totalNodes, maxLayer, numPI))

    for i in range(numPI):
        for j in range(maxLayer):

            dummy_depoStruct = (xTopo.ravel(order='F') * depoStruct[:, j, i])
            dummy_depoStruct = dummy_depoStruct.reshape((nely + subsNumlayer, nelx),order='F')

            # Element-wise material stiffness
            E_elem = (EMin + dummy_depoStruct.ravel(order='F') ** penalIHS * (EMat - EMin))
            
            
            # MATLAB:
            sK_ihs = (KE_ihs.ravel(order='F')[:, None]* E_elem[None, :]).ravel(order='F')


            # Assemble sparse stiffness matrix
            K_ihs = coo_matrix((sK_ihs,(iK_ihs.ravel(order='F') - 1,jK_ihs.ravel(order='F') - 1)),
                shape=(2 * totalNodes, 2 * totalNodes)).tocsr()
            K_ihs = (K_ihs + K_ihs.T) / 2
            
            dummy_fLayerInPI = fLayerInPI[:, j, i]

            # MATLAB DOFs -> Python 0-based
            free = freedofs_ihs - 1
            
            # Solve:
            dummy_uLayerInPI[:] = 0.0
            dummy_uLayerInPI[free] = spsolve(K_ihs[free, :][:, free],
                dummy_fLayerInPI[free]
            )
            uLayerInPI[:, j, i] = dummy_uLayerInPI
            
            # Reaction/internal force
            fAfterDepoInPI[:, j, i] = K_ihs @ dummy_uLayerInPI

    uSuper = np.sum(uLayerInPI, axis=1, keepdims=True)

    # === Objective Calculation ===
    objFunc = LN.ravel(order='F') @ np.sum(
        np.squeeze(np.sum(uLayerInPI**2, axis=1)),
        axis=1
    )
    obj_vol = np.squeeze(np.sum(eInPI, axis=0)).T
    obj_newLayer = LE.ravel(order='F') @ newLayer[:, 1, 0]
    obj_depoStruct = LE.ravel(order='F') @ depoStruct[:, 1, 0]

    # === Sensitivity Calculation ===
    # Sensitivity of the element below the PI time stamps
    deBelowPI_dt = np.zeros_like(eBelowPI)

    for i in range(numPI + 1):
        deBelowPI_dt[:, 0, i] = (-paraH1* np.exp(-paraH1 * (tPhys.ravel(order='F') - PI[i]))
            * (1.0/ (1.0+ np.exp(-paraH1 * (tPhys.ravel(order='F') - PI[i]))))** 2)

    # Sensitivity of the element in the PI
    deInPI_dt = xTopo.ravel(order='F')[:,None] * (deBelowPI_dt[:, 0, 1:] - deBelowPI_dt[:, 0, :-1])
    
    # Sensitivity of the phi term
    dphi_dt = thetaPhys * (1 + 0 * deInPI_dt).squeeze()

    
    # Sensitivity of distance
    ddistPhi_dphi = np.cos(phi) * YEC.ravel(order='F')[:,None] - np.sin(phi) * XEC.ravel(order = 'F')[:,None]
    ddistPhi_dt = ddistPhi_dphi * dphi_dt

    # sensitivity of the new layer
    dnewLayer1_ddist = np.zeros((totalEles,maxLayer,numPI))
    dnewLayer_dt = np.zeros((totalEles,maxLayer,numPI))

    for i in range(numPI):
        for j in range(maxLayer):

            dnewLayer_dt[:, j, i] = (1.0- (1.0/ (1.0+ np.exp(paraH2* (1.0- (shapeH2 / bead**2)
            * (distPhi[:, i] - (bead / 2.0) - (bead * j)) ** 2))))) * deInPI_dt[:, i]

    
    # sensitivity of the deposited structure
    ddepoStruct_dt = np.zeros((totalEles,maxLayer,numPI))

    for i in range(numPI):

        ddepoStruct_dt[:, :, i] = np.cumsum(dnewLayer_dt[:, :, i],axis=1)

        if i > 0:
            ddepoStruct_dt[:, :, i] += ddepoStruct_dt[:, -1, i - 1][:, None]

    # Lagrange multiplier calculation
    lambda_ihs = np.zeros((2*totalNodes,maxLayer,numPI))
    dummy_lambda_ihs = np.zeros((2*totalNodes,1))

    for i in range(numPI):
        for j in range(maxLayer):

            dummy_depoStruct = xTopo.ravel(order='F') * depoStruct[:, j, i]
            #dummy_depoStruct = dummy_depoStruct.reshape(nely + subsNumlayer, nelx, order='F')
            #dummy_flat = dummy_depoStruct.reshape(-1)
            
            sK_ihs = (KE_ihs.ravel(order='F')[:, None] * (EMin+ dummy_depoStruct[None,:] ** penalIHS* (EMat - EMin))).ravel(order='F')
            
            K_ihs = coo_matrix((sK_ihs,(iK_ihs.ravel(order='F')-1,jK_ihs.ravel(order='F')-1))
                               ,shape=(2*totalNodes, 2*totalNodes)).tocsr()
            K_ihs = (K_ihs + K_ihs.T) * 0.5

            dummy_uLayerInPI = uLayerInPI[:, j, i]

            free = freedofs_ihs - 1
            K_free = K_ihs[free, :][:, free]
            rhs = -2.0 * dummy_uLayerInPI[free]

            dummy_lambda_ihs[free,0] = spsolve(K_free, rhs)
            lambda_ihs[:, j, i] = dummy_lambda_ihs[:,0]

    # Total sensitvity calculation

    dKdt = np.zeros(((nely+subsNumlayer)*nelx,maxLayer,numPI))
    dFdt = np.zeros(((nely+subsNumlayer)*nelx,maxLayer,numPI))

    for i in range(numPI):
        for j in range(maxLayer):

            dummy_lambda_ihs = lambda_ihs[:, j, i]
            dummy_uLayerInPI = uLayerInPI[:, j, i]
            dummy_dnewLayer_dt = dnewLayer_dt[:, j, i]
            dummy_depoStruct = depoStruct[:, j, i]
            dummy_ddepoStruct_dt = ddepoStruct_dt[:, j, i]
            dummy_newLayer = subsFilter.ravel(order='F') * newLayer[:, j, i]

            dummy_ihsload = np.tile(ihsLoad.ravel(order='F')[None,:], (totalEles, 1))
            dummy_edofMat = (edofMat_ihs.T).ravel(order='F')
                
            dFdt[:, j, i] = (((xTopo.ravel(order='F')[:,None]) ** penalIHSf)
                * (penalIHSf * (dummy_newLayer.ravel(order='F')[:,None]) ** (penalIHSf - 1))
                * (dummy_dnewLayer_dt[:,None] * np.sum(dummy_lambda_ihs[edofMat_ihs-1] * dummy_ihsload, axis=1)[:,None])).ravel(order='F')

            if penalIHS > 1:
                dKdt[:, j, i] = (xTopo.ravel(order='F')[:,None] ** penalIHS
                    * (penalIHS * dummy_depoStruct[:,None] ** (penalIHS - 1) * (EMat - EMin))
                    * dummy_ddepoStruct_dt[:,None]
                    * (np.sum((dummy_lambda_ihs[edofMat_ihs-1] @ KE_ihs) * dummy_uLayerInPI[edofMat_ihs-1],axis=1))[:,None]).ravel(order='F')
            else:
                dKdt[:, j, i] = (xTopo.ravel(order='F')[:,None] ** penalIHS 
                    * (EMat - EMin) * dummy_ddepoStruct_dt[:,None] 
                    *(np.sum((dummy_lambda_ihs[edofMat_ihs-1] @ KE_ihs) * dummy_uLayerInPI[edofMat_ihs-1],axis=1))[:,None])

    
    dTdt = dKdt - dFdt
    sumdTdt = np.sum(dTdt, axis=(1, 2))
    sumdKdt = np.sum(dKdt, axis=(1, 2))
    sumdFdt = np.sum(dFdt, axis=(1, 2))
    
    sens_obj_vol_time = deInPI_dt
    sens_obj_newLayer_time = dnewLayer_dt[:, 1, 0]
    sens_obj_depoStruct_time = ddepoStruct_dt[:, 1, 0]


    # Sensitivity calculation of distortion - kappa variable - adjoint method

    # Initialize the Lagrange multiplier
    lambda2_T = np.zeros((totalNodes, 4))

    # Initialize the load array for adjoint
    load2_adj = np.zeros((totalNodes, 4))

    load2_adj[edofMat_T[:, 0]-1, 0] = 0.25 * sumdTdt
    load2_adj[edofMat_T[:, 1]-1, 1] = 0.25 * sumdTdt
    load2_adj[edofMat_T[:, 2]-1, 2] = 0.25 * sumdTdt
    load2_adj[edofMat_T[:, 3]-1, 3] = 0.25 * sumdTdt
    
    # Solving the adjoint equation
    lambda2_T[unknown, :] = spsolve(Et_T[unknown, :][:, unknown],load2_adj[unknown, :])
    
    # Sensitivity calculation for Et
    dummy_lambda2_T = lambda2_T[:, 0]
    Etc2_e1 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Etk2_e1 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
    dummy_lambda2_T = lambda2_T[:, 1]
    Etc2_e2 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Etk2_e2 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

    dummy_lambda2_T = lambda2_T[:, 2]
    Etc2_e3 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Etk2_e3 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

    dummy_lambda2_T = lambda2_T[:, 3]
    Etc2_e4 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Etk2_e4 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

    Etc2_e = Etc2_e1 + Etc2_e2 + Etc2_e3 + Etc2_e4
    Etk2_e = Etk2_e1 + Etk2_e2 + Etk2_e3 + Etk2_e4

    # Sensitivity calculation for At
    dummy_lambda2_T = lambda2_T[:, 0]
    Atc2_e1 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Atk2_e1 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

    dummy_lambda2_T = lambda2_T[:, 1]
    Atc2_e2 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Atk2_e2 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
    dummy_lambda2_T = lambda2_T[:, 2]
    Atc2_e3 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Atk2_e3 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

    dummy_lambda2_T = lambda2_T[:, 3]
    Atc2_e4 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    Atk2_e4 = np.sum((dummy_lambda2_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
    Atc2_e = Atc2_e1 + Atc2_e2 + Atc2_e3 + Atc2_e4
    Atk2_e = Atk2_e1 + Atk2_e2 + Atk2_e3 + Atk2_e4

    # Total Sensitivity Calculation
    Et2_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Etc2_e
        + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Etk2_e)

    At2_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Atc2_e
        + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Atk2_e)

    dobjdkappa = Et2_e + At2_e
    sens_objFuncDis_kappa = dobjdkappa

    # sensitivity of volume wrt to kappa - adjoint

    sens_vol_kappa = np.zeros((totalEles,1,numPI))

    for i in range(numPI):

        # Initialize the Lagrange multiplier
        lambda3_T = np.zeros((totalNodes, 4))

        # Initialize the load array for adjoint
        load3_adj = np.zeros((totalNodes, 4))
        load3_adj[edofMat_T[:, 0]-1, 0] = 0.25 * sens_obj_vol_time[:, i]
        load3_adj[edofMat_T[:, 1]-1, 1] = 0.25 * sens_obj_vol_time[:, i]
        load3_adj[edofMat_T[:, 2]-1, 2] = 0.25 * sens_obj_vol_time[:, i]
        load3_adj[edofMat_T[:, 3]-1, 3] = 0.25 * sens_obj_vol_time[:, i]

        # Solving the adjoint equation
        lambda3_T[unknown, :] = spsolve(Et_T[unknown, :][:, unknown],load3_adj[unknown, :])
    
        # Sensitivity calculation for Et
        dummy_lambda3_T = lambda3_T[:, 0]
        Etc3_e1 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk3_e1 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda3_T = lambda3_T[:, 1]
        Etc3_e2 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk3_e2 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda3_T = lambda3_T[:, 2]
        Etc3_e3 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk3_e3 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda3_T = lambda3_T[:, 3]
        Etc3_e4 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk3_e4 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        Etc3_e = Etc3_e1 + Etc3_e2 + Etc3_e3 + Etc3_e4
        Etk3_e = Etk3_e1 + Etk3_e2 + Etk3_e3 + Etk3_e4

        # Sensitivity calculation for At
        dummy_lambda3_T = lambda3_T[:, 0]
        Atc3_e1 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk3_e1 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda3_T = lambda3_T[:, 1]
        Atc3_e2 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk3_e2 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda3_T = lambda3_T[:, 2]
        Atc3_e3 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk3_e3 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda3_T = lambda3_T[:, 3]
        Atc3_e4 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk3_e4 = np.sum((dummy_lambda3_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        Atc3_e = Atc3_e1 + Atc3_e2 + Atc3_e3 + Atc3_e4
        Atk3_e = Atk3_e1 + Atk3_e2 + Atk3_e3 + Atk3_e4

        # Total Sensitivity Calculation
        Et3_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Etc3_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Etk3_e)

        At3_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Atc3_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Atk3_e)

        dobjdkappa = Et3_e + At3_e
        
        sens_vol_kappa[:,:,i] = dobjdkappa.ravel(order = 'F')[:,None]


    # === Gradient Calculation ===
    TN_elem = np.squeeze(TN_T_array[edofMat_T.T - 1])

    gradx_TN_T = (np.array([-1, 1, 1, -1]) @ TN_elem).T
    grady_TN_T = (np.array([-1, -1, 1, 1]) @ TN_elem).T

    # Reshape to element grid
    gradx_TN_T = gradx_TN_T.reshape((nely + subsNumlayer, nelx),order='F')
    grady_TN_T = grady_TN_T.reshape((nely + subsNumlayer, nelx),order='F')

    # Boundary another formulation

    epsBoundNew = 0.05
    kbound = 100
    epsBoundGrad = 1e-9

    bound2New = ((np.tanh(kbound * 0.5)+ np.tanh(kbound * (tPhys - 0.5)))
        / (np.tanh(kbound * 0.5)+ np.tanh(kbound * (1.0 - 0.5)))
        - (np.tanh(kbound * (0.5 + epsBoundNew))+ np.tanh(kbound * (tPhys - (0.5 + epsBoundNew))))
        / (np.tanh(kbound * (0.5 + epsBoundNew))+ np.tanh(kbound * (1.0 - (0.5 + epsBoundNew)))))
    
    grad2New = ((gradx_TN_T.ravel(order='F')[:,None] * np.cos(thetaPhys[1])
        + grady_TN_T.ravel(order='F')[:,None] * np.sin(thetaPhys[1]))).reshape(nely+subsNumlayer,nelx,order ='F')

    grad2NewMag = (gradx_TN_T**2 + grady_TN_T**2 + epsBoundGrad)**0.5
    boundGrad2 = xTopo * bound2New * (grad2New / grad2NewMag)

    # Boundary objective
    objBound2 = np.sum(boundGrad2.reshape(-1))
    objBl = np.sum(bound2New.reshape(-1))
    
    # === sensitivity of boundary wrt to kappa - adjoint

    # term - 1
    sens_grad_kappa = np.zeros((totalEles, 1, numPI - 1))

    for i in range(numPI - 1):

        # Initialize the Lagrange multiplier
        lambda4_T = np.zeros((totalNodes, 4))

        # Initialize the load array for adjoint
        load4_adj = np.zeros((totalNodes, 4))

        theta_i = thetaPhys[i + 1]

        load4_adj[edofMat_T[:, 0]-1, 0] = (xTopo.ravel(order='F')[:,None] * bound2New.ravel(order='F')[:,None] * (-1.0) 
            * ((-1.0) * ((grady_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.cos(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.sin(theta_i))
            + (-1.0) * ((gradx_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.sin(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.cos(theta_i)))
            / (grad2NewMag.ravel(order='F')[:,None]**3)).ravel(order ='F')
        load4_adj[edofMat_T[:, 1]-1, 1] = (xTopo.ravel(order='F')[:,None] * bound2New.ravel(order='F')[:,None] * (-1.0) 
            * ((1.0) * ((grady_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.cos(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.sin(theta_i))
            + (-1.0) * ((gradx_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.sin(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.cos(theta_i)))
            / (grad2NewMag.ravel(order='F')[:,None]**3)).ravel(order ='F')
        load4_adj[edofMat_T[:, 2]-1, 2] = (xTopo.ravel(order='F')[:,None] * bound2New.ravel(order='F')[:,None] * (-1.0) 
            * ((1.0) * ((grady_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.cos(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.sin(theta_i))
            + (1.0) * ((gradx_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.sin(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.cos(theta_i)))
            / (grad2NewMag.ravel(order='F')[:,None]**3)).ravel(order ='F')
        load4_adj[edofMat_T[:, 3]-1, 3] = (xTopo.ravel(order='F')[:,None] * bound2New.ravel(order='F')[:,None] * (-1.0) 
            * ((-1.0) * ((grady_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.cos(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.sin(theta_i))
            + (1.0) * ((gradx_TN_T.ravel(order='F')[:,None]**2 + epsBoundGrad) * np.sin(theta_i) 
            - (gradx_TN_T.ravel(order='F')[:,None] * grady_TN_T.ravel(order='F')[:,None]) * np.cos(theta_i)))
            / (grad2NewMag.ravel(order='F')[:,None]**3)).ravel(order ='F')
        
        # Solving the adjoint equation
        lambda4_T[unknown, :] = spsolve(Et_T[unknown, :][:, unknown],load4_adj[unknown, :])
    
        # Sensitivity calculation for Et
        dummy_lambda4_T = lambda4_T[:, 0]
        Etc4_e1 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk4_e1 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda4_T = lambda4_T[:, 1]
        Etc4_e2 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk4_e2 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda4_T = lambda4_T[:, 2]
        Etc4_e3 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk4_e3 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda4_T = lambda4_T[:, 3]
        Etc4_e4 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk4_e4 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        Etc4_e = Etc4_e1 + Etc4_e2 + Etc4_e3 + Etc4_e4
        Etk4_e = Etk4_e1 + Etk4_e2 + Etk4_e3 + Etk4_e4

        # Sensitivity calculation for At
        dummy_lambda4_T = lambda4_T[:, 0]
        Atc4_e1 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk4_e1 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda4_T = lambda4_T[:, 1]
        Atc4_e2 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk4_e2 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda4_T = lambda4_T[:, 2]
        Atc4_e3 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk4_e3 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda4_T = lambda4_T[:, 3]
        Atc4_e4 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk4_e4 = np.sum((dummy_lambda4_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        Atc4_e = Atc4_e1 + Atc4_e2 + Atc4_e3 + Atc4_e4
        Atk4_e = Atk4_e1 + Atk4_e2 + Atk4_e3 + Atk4_e4

        # Total Sensitivity Calculation
        Et4_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Etc4_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Etk4_e)

        At4_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Atc4_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Atk4_e)

        dobjdkappa = Et4_e + At4_e
        
        sens_grad_kappa[:,:,i] = dobjdkappa.ravel(order = 'F')[:,None]

    # term - 2
    sens_bound_kappa = np.zeros((totalEles,1,numPI-1))
    sens_bound_dHdT = np.zeros((totalEles,1,numPI-1))

    for i in range(numPI - 1):

        pi_val = PI[i + 1]

        sens_bound_dHdT[:, :, i] = ( kbound* ((1.0 / np.cosh(kbound * (tPhys.ravel(order = 'F') - pi_val))) ** 2
                / (np.tanh(kbound * pi_val)+ np.tanh(kbound * (1.0 - pi_val))))
                - kbound* ((1.0 / np.cosh(kbound * (tPhys.ravel(order = 'F') - (pi_val + epsBoundNew)))) ** 2
                / (np.tanh(kbound * (pi_val + epsBoundNew))+ np.tanh(kbound * (1.0 - (pi_val + epsBoundNew))))))[:,None]

    
    for i in range(numPI - 1):

        # Initialize the Lagrange multiplier
        lambda5_T = np.zeros((totalNodes, 4))

        # Initialize the load array for adjoint
        load5_adj = np.zeros((totalNodes, 4))

        load5_adj[edofMat_T[:, 0]-1, 0] = (xTopo.ravel(order='F')[:,None]* 0.25* sens_bound_dHdT[:, :, i].ravel(order='F')[:,None]
            * grad2New.ravel(order='F')[:,None]/ grad2NewMag.ravel(order='F')[:,None]).ravel(order ='F')
        load5_adj[edofMat_T[:, 1]-1, 1] = (xTopo.ravel(order='F')[:,None]* 0.25* sens_bound_dHdT[:, :, i].ravel(order='F')[:,None]
            * grad2New.ravel(order='F')[:,None]/ grad2NewMag.ravel(order='F')[:,None]).ravel(order ='F')
        load5_adj[edofMat_T[:, 2]-1, 2] = (xTopo.ravel(order='F')[:,None]* 0.25* sens_bound_dHdT[:, :, i].ravel(order='F')[:,None]
            * grad2New.ravel(order='F')[:,None]/ grad2NewMag.ravel(order='F')[:,None]).ravel(order ='F')
        load5_adj[edofMat_T[:, 3]-1, 3] = (xTopo.ravel(order='F')[:,None]* 0.25* sens_bound_dHdT[:, :, i].ravel(order='F')[:,None]
            * grad2New.ravel(order='F')[:,None]/ grad2NewMag.ravel(order='F')[:,None]).ravel(order ='F')

        # Solving the adjoint equation
        lambda5_T[unknown, :] = spsolve(Et_T[unknown, :][:, unknown],load5_adj[unknown, :])
    
        # Sensitivity calculation for Et
        dummy_lambda5_T = lambda5_T[:, 0]
        Etc5_e1 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk5_e1 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda5_T = lambda5_T[:, 1]
        Etc5_e2 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk5_e2 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda5_T = lambda5_T[:, 2]
        Etc5_e3 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk5_e3 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda5_T = lambda5_T[:, 3]
        Etc5_e4 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk5_e4 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        Etc5_e = Etc5_e1 + Etc5_e2 + Etc5_e3 + Etc5_e4
        Etk5_e = Etk5_e1 + Etk5_e2 + Etk5_e3 + Etk5_e4

        # Sensitivity calculation for At
        dummy_lambda5_T = lambda5_T[:, 0]
        Atc5_e1 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk5_e1 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda5_T = lambda5_T[:, 1]
        Atc5_e2 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk5_e2 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda5_T = lambda5_T[:, 2]
        Atc5_e3 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk5_e3 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda5_T = lambda5_T[:, 3]
        Atc5_e4 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk5_e4 = np.sum((dummy_lambda5_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        Atc5_e = Atc5_e1 + Atc5_e2 + Atc5_e3 + Atc5_e4
        Atk5_e = Atk5_e1 + Atk5_e2 + Atk5_e3 + Atk5_e4


        # Total Sensitivity Calculation
        Et5_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Etc5_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Etk5_e)

        At5_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Atc5_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Atk5_e)

        dobjdkappa = Et5_e + At5_e
        
        sens_bound_kappa[:,:,i] = dobjdkappa.ravel(order = 'F')[:,None]


    sens_gradBound_kappa = (np.squeeze(sens_grad_kappa + sens_bound_kappa))
    
    # Sensitivity of boundary 1 with respect to kappa - adjoint

    sens_bound1_kappa = np.zeros((totalEles, 1, numPI - 1))

    for i in range(numPI - 1):

        # Initialize the Lagrange multiplier
        lambda6_T = np.zeros((totalNodes, 4))

        # Initialize the load array for adjoint
        load6_adj = np.zeros((totalNodes, 4))

        load6_adj[edofMat_T[:, 0]-1, 0] = 0.25 * sens_bound_dHdT[:, :, i].ravel(order='F')
        load6_adj[edofMat_T[:, 1]-1, 1] = 0.25 * sens_bound_dHdT[:, :, i].ravel(order='F')
        load6_adj[edofMat_T[:, 2]-1, 2] = 0.25 * sens_bound_dHdT[:, :, i].ravel(order='F')
        load6_adj[edofMat_T[:, 3]-1, 3] = 0.25 * sens_bound_dHdT[:, :, i].ravel(order='F')
        
        # Solving the adjoint equation
        lambda6_T[unknown, :] = spsolve(Et_T[unknown, :][:, unknown],load6_adj[unknown, :])
    
        # Sensitivity calculation for Et
        dummy_lambda6_T = lambda6_T[:, 0]
        Etc6_e1 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk6_e1 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda6_T = lambda6_T[:, 1]
        Etc6_e2 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk6_e2 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda6_T = lambda6_T[:, 2]
        Etc6_e3 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk6_e3 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda6_T = lambda6_T[:, 3]
        Etc6_e4 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (CE_T / dt))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Etk6_e4 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * alphaEuler))* TN_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        Etc6_e = Etc6_e1 + Etc6_e2 + Etc6_e3 + Etc6_e4
        Etk6_e = Etk6_e1 + Etk6_e2 + Etk6_e3 + Etk6_e4

        # Sensitivity calculation for At
        dummy_lambda6_T = lambda6_T[:, 0]
        Atc6_e1 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk6_e1 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda6_T = lambda6_T[:, 1]
        Atc6_e2 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk6_e2 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        dummy_lambda6_T = lambda6_T[:, 2]
        Atc6_e3 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk6_e3 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')

        dummy_lambda6_T = lambda6_T[:, 3]
        Atc6_e4 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (-CE_T / dt))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
        Atk6_e4 = np.sum((dummy_lambda6_T[edofMat_T-1] @ (KE_T * (1-alphaEuler)))* TP_T_array[edofMat_T-1,0],axis=1).reshape(nely+subsNumlayer,nelx,order='F')
    
        Atc6_e = Atc6_e1 + Atc6_e2 + Atc6_e3 + Atc6_e4
        Atk6_e = Atk6_e1 + Atk6_e2 + Atk6_e3 + Atk6_e4

        # Total Sensitivity Calculation
        Et6_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Etc6_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Etk6_e)

        At6_e = (penalCp* (cpMat - cpMin)* (xTopo * xPhys_T) ** (penalCp - 1)* Atc6_e
            + penalKappa* (kappaMat - kappaMin)* (xTopo * xPhys_T) ** (penalKappa - 1)* Atk6_e)

        dobjdkappa = Et6_e + At6_e
        sens_bound1_kappa[:,:,i] = dobjdkappa.ravel(order = 'F')[:,None]

    #=== Sensitivity with respect to theta - finite difference

    # Initialize the variable for finite difference analysis
    thetaPhys_fd2 = thetaPhys.copy()

    # Initialize the sensitivity variables
    sens_objFuncDis_theta_fd2 = np.zeros((1, numPI))
    sens_consbound1_theta_fd2 = np.zeros((1, numPI))
    sens_consbound2_theta_fd2 = np.zeros((1, numPI))
    sens_consbound3_theta_fd2 = np.zeros((1, numPI))

    # Initialize perturbation
    perturb = 0.1
    
    for i in range(thetaPhys.size):

        # Perturbation to the design variable theta
        thetaPhys_fd2[i] = thetaPhys_fd2[i] + perturb

        # Time design variable
        tPhys_fd2 = tPhys.copy()

        # Initialize array to identify elements below the time stamp
        eBelowPI_fd2 = np.zeros((totalEles, 1, numPI + 1))

        for ii in range(1, numPI + 1):
            eBelowPI_fd2[:, 0, ii] = (1.0- 1.0 / (1.0+ np.exp(-paraH1
                    * (tPhys_fd2.ravel(order = 'F') - PI[ii]))))

        eInPI_fd2 = (eBelowPI_fd2[:, 0, 1:]- eBelowPI_fd2[:, 0, :-1])
        
        totalEInPI_fd2 = np.sum(eInPI_fd2, axis=0)
        
        phi_fd2 = thetaPhys_fd2 * np.squeeze(1 + 0 * eInPI_fd2)
        

        distPhi_fd2 = (np.sin(phi_fd2) * YEC.ravel(order = 'F')[:,None]
            + np.cos(phi_fd2) * XEC.ravel(order = 'F')[:,None])

        newLayer_fd2 = np.zeros((totalEles, maxLayer, numPI))

        for ii in range(numPI):
            for jj in range(maxLayer):

                newLayer_fd2[:, jj, ii] = (eInPI_fd2[:, ii]
                    * (1.0- 1.0 / (1.0+ np.exp(paraH2* (1.0- (shapeH2 / bead**2)
                    * (distPhi_fd2[:, ii]- bead / 2.0- bead * jj) ** 2)))))
        
        depoStruct_fd2 = np.cumsum(newLayer_fd2, axis=1)

        for ii in range(1, numPI):
            depoStruct_fd2[:, :, ii] += depoStruct_fd2[:, -1, ii - 1][:, None]

        subFilterIndexMat = np.where(subsFilter.ravel(order='F') == 0)[0]
        depoStruct_fd2[subFilterIndexMat, :, :] = 1

        # === Force calculation ===
        fLayerInPI_fd2 = np.zeros((2 * totalNodes, maxLayer, numPI))

        for ii in range(numPI):
            for jj in range(maxLayer):

                dummy_newLayer_fd2 = (xTopo.ravel(order='F')* subsFilter.ravel(order='F')* newLayer_fd2[:, jj, ii])
            
                if penalIHSf != 0:
                    dummy_newLayer_kron_fd2 = np.kron(dummy_newLayer_fd2 ** penalIHSf,np.ones(ihsLoad.shape[0]))
                else:
                    dummy_newLayer_kron_fd2 = np.kron(1 + 0 * dummy_newLayer_fd2,np.ones(ihsLoad.shape[0]))

                dummy_ihsload_fd2 = np.tile(ihsLoad, (totalEles, 1)).ravel()
                dummy_edofMat_fd2 = edofMat_ihs.ravel()
            
                # MATLAB sparse() uses 1-based indices; SciPy uses 0-based
                dummy_fLayerInPI_fd2 = coo_matrix(( dummy_newLayer_kron_fd2 * dummy_ihsload_fd2.ravel(order='F'),
                        (dummy_edofMat_fd2 - 1, np.zeros(dummy_edofMat_fd2.size, dtype=int))),
                    shape=(2 * totalNodes, 1)).toarray().ravel()
            
                fLayerInPI_fd2[:, jj, ii] = dummy_fLayerInPI_fd2
        
        # === Finite Element - Displacement Calculation ===
        uLayerInPI_fd2 = np.zeros((2 * totalNodes, maxLayer, numPI))
        dummy_uLayerInPI_fd2 = np.zeros(2 * totalNodes)
        
        for ii in range(numPI):
            for jj in range(maxLayer):

                dummy_depoStruct_fd2 = (xTopo.ravel(order='F') * depoStruct_fd2[:, jj, ii])
                dummy_depoStruct_fd2 = dummy_depoStruct_fd2.reshape((nely + subsNumlayer, nelx),order='F')

                # Element-wise material stiffness
                E_elem_fd2 = (EMin + dummy_depoStruct_fd2.ravel(order='F') ** penalIHS * (EMat - EMin))
            
                # MATLAB:
                sK_ihs_fd2 = (KE_ihs.ravel(order='F')[:, None]* E_elem_fd2[None, :]).ravel(order='F')

                # Assemble sparse stiffness matrix
                K_ihs_fd2 = coo_matrix((sK_ihs_fd2,(iK_ihs.ravel(order='F') - 1,jK_ihs.ravel(order='F') - 1)),
                    shape=(2 * totalNodes, 2 * totalNodes)).tocsr()
                K_ihs_fd2 = (K_ihs_fd2 + K_ihs_fd2.T) / 2
            
                dummy_fLayerInPI_fd2 = fLayerInPI_fd2[:, jj, ii]

                # MATLAB DOFs -> Python 0-based
                free = freedofs_ihs - 1
            
                # Solve:
                dummy_uLayerInPI_fd2[:] = 0.0
                dummy_uLayerInPI_fd2[free] = spsolve(K_ihs_fd2[free, :][:, free],
                    dummy_fLayerInPI_fd2[free]
                )
                uLayerInPI_fd2[:, jj, ii] = dummy_uLayerInPI_fd2
        
        uSuper_fd2 = np.sum(uLayerInPI_fd2, axis=1, keepdims=True)
        
        objFunc_fd2 = LN.ravel(order='F') @ np.sum(
        np.squeeze(np.sum(uLayerInPI_fd2**2, axis=1)),
        axis=1)

        sens_objFuncDis_theta_fd2[0, i] = (objFunc_fd2 - objFunc) / perturb

        bound2New_fd2 = ((np.tanh(kbound * 0.5)+ np.tanh(kbound * (tPhys_fd2 - 0.5)))
            / (np.tanh(kbound * 0.5)+ np.tanh(kbound * (1.0 - 0.5)))
            - (np.tanh(kbound * (0.5 + epsBoundNew))+ np.tanh(kbound * (tPhys_fd2 - (0.5 + epsBoundNew))))
            / (np.tanh(kbound * (0.5 + epsBoundNew))+ np.tanh(kbound * (1.0 - (0.5 + epsBoundNew)))))
    
        grad2New_fd2 = ((gradx_TN_T.ravel(order='F')[:,None] * np.cos(thetaPhys_fd2[1])
        + grady_TN_T.ravel(order='F')[:,None] * np.sin(thetaPhys_fd2[1]))).reshape(nely+subsNumlayer,nelx,order ='F')

        grad2NewMag_fd2 = (gradx_TN_T**2 + grady_TN_T**2 + epsBoundGrad)**0.5
        boundGrad2_fd2 = xTopo * bound2New_fd2 * (grad2New_fd2 / grad2NewMag_fd2)

        # Boundary objective
        objBound2_fd2 = np.sum(boundGrad2_fd2.reshape(-1))
        objBl_fd2 = np.sum(bound2New_fd2.reshape(-1))

        sens_consbound2_theta_fd2[0, i] = (objBound2_fd2 - objBound2) / perturb

        thetaPhys_fd2[i] = thetaPhys_fd2[i] - perturb

    # Optimization problem definition

    weightDis = 800  # abs(objFunc)
    weightBound = 1  # abs(objBound2)

    f0val = (100.0 * (objFunc / weightDis) + 1.0 * ((objBound2 + objBl) / weightBound))
    df0dx = (100.0* (np.concatenate([sens_objFuncDis_kappa.ravel(order='F'),sens_objFuncDis_theta_fd2.ravel(order='F')])/ weightDis)
        + 1.0* (np.concatenate([(sens_gradBound_kappa.ravel(order='F') + sens_bound1_kappa.ravel(order='F')),sens_consbound2_theta_fd2.ravel(order='F')])/ weightBound))[:,None]
    df0dx2 = 0.0 * df0dx

    # Volume constraint
    fval01 = ((obj_vol / (np.sum(curve) + np.sum(subs)))/ (1.0 / numPI)- 1.0)
    df01dx = (np.vstack([np.squeeze(sens_vol_kappa),np.zeros((numPI, numPI))])
              / (np.sum(curve) + np.sum(subs))) / (1.0 / numPI)
    df01dx2 = 0.0 * df01dx
    
    if loop % 20 == 0 and loop <= 100:
        consBoundEps /= 10

    fval = fval01.ravel(order = 'F')[:,None]
    dfdx = df01dx.T
    dfdx2 = 0 * dfdx

    # == To the optimizer ===

    xval = np.concatenate([
        xPhys_T.ravel(order = 'F'),
        thetaPhys.ravel(order = 'F')
    ])[:,None]

    xmax = np.concatenate([
        np.minimum(1, xPhys_T.ravel(order = 'F') + movekappa),
        np.minimum(np.deg2rad(90), thetaPhys.ravel(order = 'F') + movetheta)
    ])[:,None]

    xmin = np.concatenate([
        np.maximum(0, xPhys_T.ravel(order= 'F') - movekappa),
        np.maximum(np.deg2rad(0), thetaPhys.ravel(order = 'F') - movetheta)
    ])[:,None]
    
    xmma, *_, low, upp = mmasub(
        m, n, loop, xval, xmin, xmax, xold1, xold2,
        f0val, df0dx, df0dx2, fval, dfdx, dfdx2,low, upp, a0mma, amma, cmma, dmma)

    xold2 = xold1
    xold1 = xval
    xnew = xmma

    xPhys_T_new = xnew[:-numPI]
    thetaPhys_new = xnew[-numPI:]
    
    subsIndex = np.where(subsFilter.ravel(order='F') == 0)[0]

    xPhys_T = xPhys_T_new.reshape(nely + subsNumlayer, nelx, order='F')
    xPhys_T = (H_C @ xPhys_T.ravel(order='F')[:,None]) / Hs_C
    xPhys_T = xPhys_T.reshape(nely + subsNumlayer, nelx, order='F')

    xPhys_T_visual = xTopo * xPhys_T
    
    xPhys_T = xPhys_T.ravel(order='F')
    xPhys_T[subsIndex] = 1
    xPhys_T = xPhys_T.reshape(nely + subsNumlayer,nelx,order='F')

    thetaPhys = (thetaPhys_new.T).ravel(order='F')

    xPhys_T_old = xval[:len(xval) - numPI]
    thetaPhys_old = xval[len(xval) - numPI:]

    changekappa = np.max(np.abs(xPhys_T_new.ravel() - xPhys_T_old.ravel()))
    changetheta = np.max(np.abs(thetaPhys_new.ravel() - thetaPhys_old.ravel()))

    objTrack = np.vstack([objTrack,np.concatenate([
        np.array([loop,f0val,objFunc,np.sum(bound2New),objBound2,]),
        np.asarray(fval).ravel(),
        np.asarray(thetaPhys).ravel() * (180 / np.pi),
        np.array([changekappa,changetheta])])])
    
    print(
    f"loop: {loop:d}, "
    f"Obj: {f0val:4.4f}, "
    f"disObj: {objFunc:4.4f}, "
    f"boundObj: {objBound2:4.4f}, "
    f"totalBound: {np.sum(bound2New):4.4f}, "
    f"theta1: {180 / np.pi * thetaPhys}"
    )

    workspace = {
        "uSuper": uSuper,
        "xTopo": xTopo,
        "subsFilter": subsFilter,
        "edofMat_ihs": edofMat_ihs,
        "edofMat_T": edofMat_T,
        "XNC": XNC,
        "YNC": YNC,
        "XEC": XEC,
        "YEC": YEC,
        "nelx": nelx,
        "nely": nely,
        "subsNumlayer": subsNumlayer,
        "numPI": numPI ,
        "thetaPhys" : thetaPhys,
        "bead" :bead,
        "maxLayer": maxLayer,
        "tPhys" : tPhys,
        "PI" : PI
        }

    if loop == 1 or loop % 50 == 0:
        filename = f"files{loop}.pkl"
        #sio.savemat(filename, {
        #    "objTrack": objTrack,
        #    "uSuper" : uSuper,
        #    "nely" : nely,
        #    "nelx" : nelx,
        #    "subsNumlayer" : subsNumlayer,
        #    "xTopo" : xTopo,
        #    "subsFilter" : subsFilter,
        #    "edofMat_ihs" : edofMat_ihs,
        #    "XNC" : XNC,
        #    "YNC" : YNC
        #})

        with open(filename, "wb") as f:
            pickle.dump(workspace, f)



    if loop == 100:
        filename = f"files{loop}.pkl"
        #sio.savemat(filename, {
        #    "objTrack": objTrack,
        #    "uSuper" : uSuper,
        #    "nely" : nely,
        #    "nelx" : nelx,
        #    "subsNumlayer" : subsNumlayer,
        #    "xTopo" : xTopo,
        #    "subsFilter" : subsFilter,
        #    "edofMat_ihs" : edofMat_ihs,
        #    "XNC" : XNC,
        #    "YNC" : YNC
        #})
        with open(filename, "wb") as f:
            pickle.dump(workspace, f)








    
    

    
            



