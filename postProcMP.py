import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pickle
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.validation import make_valid

with open("files1.pkl", "rb") as f:
    workspace = pickle.load(f)

globals().update(workspace)


# clear tPhysPI
tPhys_visual = xTopo * tPhys
tPhys_visual[-1, :] = -1
tPhys_visual = tPhys_visual.ravel(order='F')

tPhysPI = [dict() for _ in range(numPI)]
PIIndex = []

for ii in range(numPI):

    # MATLAB:
    # find(tPhys_visual(:)>PI(ii,1) & tPhys_visual(:)<=PI(ii+1,1))
    mask = (
        (tPhys_visual.ravel(order = 'F') > PI[ii])
        & (tPhys_visual.ravel(order = 'F') <= PI[ii + 1])
    )

    # MATLAB find() returns 1-based indices
    # Python indices are 0-based, so no +1 is needed here
    tPhysPI[ii]["PIIndex"] = np.where(mask)[0]

    # MATLAB: edofMat_T(tPhysPI(ii).PIIndex,:)
    tPhysPI[ii]["PI"] = edofMat_T[tPhysPI[ii]["PIIndex"], :]

    # MATLAB: unique(tPhysPI(ii).PI(:))
    tPhysPI[ii]["PINodeNum"] = np.unique(tPhysPI[ii]["PI"].ravel(order = 'F'))

    # MATLAB: XNC(tPhysPI(ii).PINodeNum)
    tPhysPI[ii]["PIX"] = XNC.ravel(order = 'F')[tPhysPI[ii]["PINodeNum"]]
    tPhysPI[ii]["PIY"] = YNC.ravel(order = 'F')[tPhysPI[ii]["PINodeNum"]]

    # Element centroid coordinates
    tPhysPI[ii]["PIEX"] = XEC.ravel(order = 'F')[tPhysPI[ii]["PIIndex"]]
    tPhysPI[ii]["PIEY"] = YEC.ravel(order = 'F')[tPhysPI[ii]["PIIndex"]]

    # MATLAB:
    # PIIndex = [PIIndex; tPhysPI(ii).PIIndex];
    PIIndex.extend(tPhysPI[ii]["PIIndex"].tolist())

# Convert accumulated indices to NumPy array
PIIndex = np.asarray(PIIndex, dtype=int)

# Common nodes between consecutive process intervals
for ii in range(numPI - 1):

    tPhysPI[ii]["PICommonNodeNum"] = np.intersect1d(tPhysPI[ii]["PINodeNum"],tPhysPI[ii + 1]["PINodeNum"])
    tPhysPI[ii]["PICommonNodeNumX"] = (XNC.ravel(order = 'F')[tPhysPI[ii]["PICommonNodeNum"]])
    tPhysPI[ii]["PICommonNodeNumY"] = (YNC.ravel(order = 'F')[tPhysPI[ii]["PICommonNodeNum"]])

for ii, brakcets in enumerate(tPhysPI):
    print(f"\nBracket[{ii}]:")
    for key, value in brakcets.items():
        print(f"  {key}: {value}")

# %% Polygon formation

# ---------------------------------------------------------
# Total polygon
# ---------------------------------------------------------

dummyNodeNum = edofMat_T[PIIndex[0], :]

dummyXC = XNC.ravel(order = 'F')[dummyNodeNum]
dummyYC = YNC.ravel(order = 'F')[dummyNodeNum]

dummyPolyshape = Polygon(
    np.column_stack((dummyXC, dummyYC))
)

polygonTotal = dummyPolyshape

polygonPI = [None] * numPI

for ii in range(1, len(PIIndex)):

    dummyNodeNum = edofMat_T[PIIndex[ii], :]

    dummyXC = XNC.ravel(order = 'F')[dummyNodeNum]
    dummyYC = YNC.ravel(order = 'F')[dummyNodeNum]

    dummyPolyshape = Polygon(np.column_stack((dummyXC, dummyYC)))
    # MATLAB:
    # dummyUnion = union(dummyPolyshape, polygonTotal);
    dummyUnion = dummyPolyshape.union(polygonTotal)

    polygonTotal = dummyUnion

for ii in range(numPI):

    # First element associated with this PI
    dummyNodeNum = edofMat_T[tPhysPI[ii]["PIIndex"][0], :]

    # MATLAB uses 1-based node numbers
    dummyNodeNum = dummyNodeNum.astype(int) - 1


    dummyXC = XNC.ravel(order='F')[dummyNodeNum]
    dummyYC = YNC.ravel(order='F')[dummyNodeNum]

    dummyPolyshape = Polygon(
        np.column_stack((dummyXC, dummyYC))
    )

    polygonPI[ii] = {"PI": dummyPolyshape}

    # Remaining elements
    for jj in range(1, len(tPhysPI[ii]["PIIndex"])):

        dummyNodeNum = edofMat_T[
            tPhysPI[ii]["PIIndex"][jj], :
        ]

        dummyNodeNum = dummyNodeNum.astype(int) - 1

        dummyXC = XNC.ravel(order='F')[dummyNodeNum]
        dummyYC = YNC.ravel(order='F')[dummyNodeNum]

        dummyPolyshape = Polygon(
            np.column_stack((dummyXC, dummyYC))
        )

        dummyUnion = dummyPolyshape.union(
            polygonPI[ii]["PI"]
        )

        polygonPI[ii]["PI"] = dummyUnion


plt.figure(1)
plt.clf()

for ii in range(numPI):
    polygon = polygonPI[ii]["PI"]

    if polygon.geom_type == "Polygon":
        x, y = polygon.exterior.xy
        plt.fill(x, y, alpha=0.5)
        plt.plot(x, y, linewidth=4)

    elif polygon.geom_type == "MultiPolygon":
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.5)
            plt.plot(x, y, linewidth=4)

#plt.axis("off")
#plt.autoscale(enable=True, axis="both", tight=True)
plt.axis("tight")
plt.axis("equal")


# ==== LINE CALCUALTION =====

domainSize = max(nelx, nely)

# Clear Line
Line = [dict() for _ in range(numPI)]

for ii in range(numPI):

    theta = thetaPhys[ii]

    if theta >= 0 and theta <= np.pi / 2:

        maxLayer = int(
            np.ceil(
                (
                    np.cos(theta) * (domainSize + subsNumlayer)
                    + np.sin(theta) * domainSize
                ) / bead
            )
        )

        Line[ii]["maxLayer"] = maxLayer

        layer_values = np.arange(maxLayer + 1)

        Line[ii]["InterceptYaxis"] = (
            bead * layer_values / np.sin(theta)
        )

        Line[ii]["InterceptXaxis"] = (
            bead * layer_values / np.cos(theta)
        )

        Line[ii]["InterceptUB"] = (
            (bead * layer_values / np.sin(theta))
            * np.tan(theta)
            - (domainSize + subsNumlayer) * np.tan(theta)
        )

        Line[ii]["InterceptRB"] = (
            (bead * layer_values / np.sin(theta))
            - domainSize / np.tan(theta)
        )

    if theta < 0 and theta >= -np.pi / 2:

        maxLayer1 = int(np.ceil((np.cos(theta) * (domainSize + subsNumlayer)+ np.sin(theta) * 0) / bead))
        maxLayer2 = int(np.floor((np.cos(theta) * 0 + np.sin(theta) * domainSize) / bead)- 2)

        maxLayer = abs(maxLayer1) + abs(maxLayer2)

        Line[ii]["maxLayer"] = maxLayer

        layer_values = np.arange(maxLayer2, maxLayer1 + 1)

        Line[ii]["InterceptYaxis"] = (bead * layer_values / np.sin(theta))
        Line[ii]["InterceptXaxis"] = (bead * layer_values / np.cos(theta))
        Line[ii]["InterceptUB"] = ((bead * layer_values / np.sin(theta))* np.tan(theta)- (domainSize + subsNumlayer) * np.tan(theta))
        Line[ii]["InterceptRB"] = ((bead * layer_values / np.sin(theta))- domainSize / np.tan(theta))


# %% Calculate line end points

for ii in range(numPI):

    maxLayer = Line[ii]["maxLayer"]

    # MATLAB: jj = 2:Line(ii).maxLayer
    for jj in range(1, maxLayer):

        theta = thetaPhys[ii]

        # Python index corresponding to MATLAB jj
        idx = jj

        if theta >= 0 and theta <= np.pi / 2:

            if (Line[ii]["InterceptYaxis"][idx]<= domainSize + subsNumlayer
                and
                Line[ii]["InterceptXaxis"][idx]<= domainSize
            ):

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptXaxis"][idx]
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = 0
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = 0
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptYaxis"][idx]

            elif (Line[ii]["InterceptYaxis"][idx]> domainSize + subsNumlayer
                and
                Line[ii]["InterceptXaxis"][idx]<= domainSize
            ):

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptXaxis"][idx]
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = 0
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptUB"][idx]
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = domainSize + subsNumlayer

            elif (Line[ii]["InterceptYaxis"][idx]> domainSize + subsNumlayer
                and
                Line[ii]["InterceptXaxis"][idx]> domainSize
            ):

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = domainSize
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptRB"][idx]
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptUB"][idx]
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = domainSize + subsNumlayer

            else:

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = domainSize
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptRB"][idx]
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = 0
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptYaxis"][idx]

        # ---------------------------------------------------------
        # Negative theta
        # ---------------------------------------------------------
        if theta < 0 and theta >= -np.pi / 2:

            if (
                Line[ii]["InterceptYaxis"][idx]<= domainSize + subsNumlayer
                and
                Line[ii]["InterceptUB"][idx]<= domainSize
                and
                Line[ii]["InterceptXaxis"][idx]<= 0
            ):

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptUB"][idx]
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = domainSize + subsNumlayer
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = 0
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptYaxis"][idx]

            elif (
                Line[ii]["InterceptYaxis"][idx]<= domainSize + subsNumlayer
                and
                Line[ii]["InterceptUB"][idx]> domainSize
                and
                Line[ii]["InterceptXaxis"][idx]<= 0
            ):

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = domainSize
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptRB"][idx]
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = 0
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptYaxis"][idx]

            elif (
                Line[ii]["InterceptYaxis"][idx] < 0
                and
                Line[ii]["InterceptXaxis"][idx] > 0
                and
                Line[ii]["InterceptUB"][idx] <= domainSize
            ):

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptUB"][idx]
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = domainSize + subsNumlayer
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptXaxis"][idx]
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = 0

            else:

                Line[ii].setdefault("Point1X",np.zeros(maxLayer - 1))[jj - 1] = nelx
                Line[ii].setdefault("Point1Y",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptRB"][idx]
                Line[ii].setdefault("Point2X",np.zeros(maxLayer - 1))[jj - 1] = Line[ii]["InterceptXaxis"][idx]
                Line[ii].setdefault("Point2Y",np.zeros(maxLayer - 1))[jj - 1] = 0


for ii, line in enumerate(Line):
    print(f"\nLine[{ii}]:")
    for key, value in line.items():
        print(f"  {key}: {value}")

for ii, line in enumerate(polygonPI):
    print(f"\nLine[{ii}]:")
    for key, value in line.items():
        print(f"  {key}: {value}")


# Figure 12
fig = plt.figure(1)
ax = fig.add_subplot()

for ii in range(numPI):

    # MATLAB: 1:Line(ii).maxLayer-1
    for jj in range(Line[ii]["maxLayer"] - 2 + 1):

        # ---------------------------------------------------------
        # Line geometry
        # ---------------------------------------------------------
        Line[ii].setdefault("dy",np.zeros(maxLayer - 1))[jj] = (
            Line[ii]["Point2Y"][jj] - Line[ii]["Point1Y"][jj]
        )

        Line[ii].setdefault("dx",np.zeros(maxLayer - 1))[jj] = (
            Line[ii]["Point2X"][jj] - Line[ii]["Point1X"][jj]
        )

        Line[ii].setdefault("slope",np.zeros(maxLayer - 1))[jj] = (
            Line[ii]["Point2Y"][jj] - Line[ii]["Point1Y"][jj]
        ) / (
            Line[ii]["Point2X"][jj] - Line[ii]["Point1X"][jj]
        )

        # ---------------------------------------------------------
        # Distance of points from line
        # ---------------------------------------------------------
        slope = Line[ii]["slope"][jj]

        PIX = np.asarray(tPhysPI[ii]["PIX"]).ravel(order = 'F')
        PIY = np.asarray(tPhysPI[ii]["PIY"]).ravel(order = 'F')

        LineDist = np.abs(
            (
                (PIY - Line[ii]["Point1Y"][jj])
                - slope * (PIX - Line[ii]["Point1X"][jj])
            )
            / np.sqrt(1.0 + slope**2)
        )

        # MATLAB:
        # find(LineDist(:)>=0 & LineDist(:)<=100*bead)
        LineDistBeadIndex = np.where(
            (LineDist >= 0) & (LineDist <= 100 * bead)
        )[0]

        # ---------------------------------------------------------
        # Define line points
        # ---------------------------------------------------------
        LinePoint1 = np.array([
            Line[ii]["Point1X"][jj],
            Line[ii]["Point1Y"][jj]
        ])

        LinePoint2 = np.array([
            Line[ii]["Point2X"][jj],
            Line[ii]["Point2Y"][jj]
        ])

        VectorLine = LinePoint2 - LinePoint1

        DotVectorLine = np.sum(VectorLine * VectorLine)

        # ---------------------------------------------------------
        # Points inside bead region
        # ---------------------------------------------------------
        PointsInBead = np.column_stack([
            PIX[LineDistBeadIndex],
            PIY[LineDistBeadIndex]
        ])

        VectorPointsInBead = PointsInBead - LinePoint1

        DotVectorPointsInBead = np.sum(
            VectorPointsInBead * VectorLine,
            axis=1
        )

        # ---------------------------------------------------------
        # Projection of points onto line
        # ---------------------------------------------------------
        Projection = (
            LinePoint1
            + (DotVectorPointsInBead / DotVectorLine)[:, np.newaxis]
            * VectorLine
        )


        # ---------------------------------------------------------
        # inpolygon equivalent
        # ---------------------------------------------------------
        vertices = np.column_stack([
            np.asarray(polygonPI[ii]["PI"].exterior.coords)[:, 0],
            np.asarray(polygonPI[ii]["PI"].exterior.coords)[:, 1]
        ])

        polygon_path = Path(vertices)

        ProjPointsInPoly = polygon_path.contains_points(Projection)

        # MATLAB:
        # ProjFilter = Projection(find(ProjPointsInPoly==1),:)
        ProjFilter = Projection[ProjPointsInPoly]

        # ---------------------------------------------------------
        # Distance from LinePoint1
        # ---------------------------------------------------------
        ProjFilterDist = np.sqrt(
            np.sum((ProjFilter - LinePoint1)**2, axis=1)
        )

        # MATLAB:
        # [ProjFilter, ProjFilterDist]
        ProjFilterDistComb = np.column_stack([
            ProjFilter,
            ProjFilterDist
        ])

        # Sort according to 3rd column
        ProjFilterDistCombSort = (
            ProjFilterDistComb[
                np.argsort(ProjFilterDistComb[:, 2])
            ]
        )

        # MATLAB: 1:5:end
        ProjFilterDistCombSort = ProjFilterDistCombSort[::5]

        # ---------------------------------------------------------
        # Check whether we have points
        # ---------------------------------------------------------
        if ProjFilterDistCombSort.shape[0] > 0:

            # MATLAB:
            # [0:0.01:1]'
            s = np.arange(0.0, 1.0 + 0.01, 0.01)

            midPointLine = (
                s[:, np.newaxis]
                * ProjFilterDistCombSort[0, 0:2]
                + (1.0 - s[:, np.newaxis])
                * ProjFilterDistCombSort[-1, 0:2]
            )

            midPointInPoly = polygon_path.contains_points(
                midPointLine
            )

            # -----------------------------------------------------
            # Entire line segment is inside polygon
            # -----------------------------------------------------
            if np.sum(midPointInPoly) == midPointLine.shape[0]:

                ax.plot(
                    ProjFilterDistCombSort[[0, -1], 0],
                    ProjFilterDistCombSort[[0, -1], 1],
                    'k',
                    linewidth=2
                )

            # -----------------------------------------------------
            # Only some portions are inside polygon
            # -----------------------------------------------------
            else:

                for kk in range(
                    ProjFilterDistCombSort.shape[0] - 1
                ):

                    midPointLine = 0.5 * (
                        ProjFilterDistCombSort[kk, 0:2]
                        + ProjFilterDistCombSort[kk + 1, 0:2]
                    )

                    midPointInPoly = polygon_path.contains_points(
                        midPointLine.reshape(1, 2)
                    )[0]

                    if midPointInPoly:

                        ax.plot(
                            ProjFilterDistCombSort[
                                kk:kk + 2, 0
                            ],
                            ProjFilterDistCombSort[
                                kk:kk + 2, 1
                            ],
                            'k',
                            linewidth=2
                        )


# -------------------------------------------------------------
# Final figure formatting
# -------------------------------------------------------------
ax.axis('tight')
ax.set_aspect('equal')
ax.axis('off')
plt.show()