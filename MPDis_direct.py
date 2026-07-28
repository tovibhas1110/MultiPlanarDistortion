import math
import os  # os module - talking to the system to create files and folders
import time  # time module - time related functions and operations
from functools import partial
from typing import NamedTuple

import gmsh  # gmsh - Python API - giving access to the finite element mesh generator
import jax  # Jax - google's high performance numerical computing library - numpy on steroid
import jax.numpy as jnp  # Implements the NumPy API, using the primitives in jax.lax.
import matplotlib.pyplot as plt  # It is a collection of functions that make matplotlib work like MATLAB.
import meshio  # open-source Python library used for reading, writing, and converting unstructured mesh data between various file formats.
import numpy as np  # It is a fundamental open-source library in Python used for scientific computing, data analysis, and machine learning.
from jax import Array  # JAX array class
from jax_autovmap import (
    autovmap,  # Autovmap automatically vectorizes a function over batched inputs using JAX's vmap.
)
from matplotlib.axes import (
    Axes,  # An Axes object represents the actual plotting area of a figure.
)
from scipy.sparse.linalg import splu  # sparse LU factorization of the tangent system
from tatva import (  # tatva FEM building blocks
    Mesh,
    Operator,
    element,
    sparse,  # graph-coloring sparse Jacobian assembly
)
from tatva.lifter import (  # Dirichlet BCs via reduction to free DOFs
    Fixed,
    Lifter,
)

jax.config.update(
    "jax_enable_x64", True
)  # Updating the configuration of JAX to high-precision


# function definition plot_mesh
def plot_mesh(mesh: Mesh, color, ax: Axes | None = None) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    tpc = ax.tripcolor(
        mesh.coords[:, 0],
        mesh.coords[:, 1],
        mesh.elements,
        facecolors=color,
        cmap="managua",
        edgecolors="k",
        linewidth=0.2,
        vmin=0,
        vmax=1,
    )
    ax.set_aspect("equal")
    plt.colorbar(tpc, ax=ax)
    return ax


import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_force_vectors(mesh, f_ext, normalize=True, scale=1, figsize=(6, 6)):
    """
    Plot nodal force vectors using a quiver plot.

    Parameters
    ----------
    mesh : object
        Mesh object containing `coords` of shape (n_nodes, 2).
    f_ext : array_like
        External force vector of length 2*n_nodes arranged as
        [Fx1, Fy1, Fx2, Fy2, ...].
    normalize : bool, optional
        If True, plot unit vectors. If False, plot actual force vectors.
    scale : float, optional
        Quiver scale parameter.
    figsize : tuple, optional
        Figure size.
    """

    fx = jnp.asarray(f_ext[0::2])
    fy = jnp.asarray(f_ext[1::2])

    if normalize:
        norm = jnp.sqrt(fx**2 + fy**2)
        norm = jnp.where(norm == 0, 1.0, norm)
        u = fx / norm
        v = fy / norm
    else:
        u = fx
        v = fy

    plt.figure(figsize=figsize)
    plt.quiver(
        mesh.coords[:, 0],
        mesh.coords[:, 1],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=scale,
    )

    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.xlim(mesh.coords[:, 0].min() - 1, mesh.coords[:, 0].max() + 1)
    plt.ylim(mesh.coords[:, 1].min() - 1, mesh.coords[:, 1].max() + 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Definition of the geometry and mesh as output
def generate_refined_plate(
    width: float,
    height: float,
    subsheight: float,
    mesh_size_fine: float,
    mesh_size_coarse: float,
):
    mesh_dir = os.path.join(os.getcwd(), "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    output_filename = os.path.join(mesh_dir, "plate_refined.msh")

    gmsh.initialize()
    gmsh.model.add("plate_refined")
    occ = gmsh.model.occ
    # First rectangle
    subs = occ.addRectangle(0, -subsheight, 0, width, subsheight)
    # Second rectangle directly above the first
    # Trapezoid parameters
    top_width = 2 * width  # Width of the top edge
    offset = (width - top_width) / 2

    # Corner points (counter-clockwise)
    p1 = occ.addPoint(0, 0, 0)
    p2 = occ.addPoint(width, 0, 0)
    p3 = occ.addPoint(width - offset, height, 0)
    p4 = occ.addPoint(0, height, 0)

    # Edges
    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p4)
    l4 = occ.addLine(p4, p1)

    # Surface
    loop = occ.addCurveLoop([l1, l2, l3, l4])
    rect = occ.addPlaneSurface([loop])
    out, _ = occ.fragment([(2, rect)], [(2, subs)])
    occ.synchronize()

    surface_tags = [s[1] for s in out]
    gmsh.model.addPhysicalGroup(2, surface_tags, 1, name="domain")

    boundaries = gmsh.model.getBoundary(out, oriented=False)
    boundary_tags = [b[1] for b in boundaries]
    gmsh.model.addPhysicalGroup(1, boundary_tags, 2, name="boundaries")

    curve_to_surfaces = {}
    for c in gmsh.model.getEntities(1):
        adj = gmsh.model.getAdjacencies(1, c[1])[1]
        curve_to_surfaces[c[1]] = set(adj)

    interface_curves = [c for c, adj in curve_to_surfaces.items() if len(adj) == 2]
    gmsh.model.addPhysicalGroup(1, interface_curves, 3, name="interface")

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.model.mesh.setTransfiniteCurve(rect, 100)
    gmsh.model.mesh.setTransfiniteCurve(subs, 100)

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", interface_curves)
    gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size_fine)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size_coarse)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", height * 2.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()

    _mesh = meshio.read(output_filename)
    coords = _mesh.points[:, :2]
    elements = _mesh.cells_dict["triangle"]

    return Mesh(coords=coords, elements=elements)


def deformMesh(mesh: Mesh, u):

    ux = u[0::2]
    uy = u[1::2]

    deformCoordx = mesh.coords[:, 0] + ux
    deformCoordy = mesh.coords[:, 1] + uy

    deformCoords = jnp.column_stack((deformCoordx, deformCoordy))

    elements = mesh.elements

    return Mesh(coords=deformCoords, elements=elements)


t_start = time.perf_counter()

lx = 100.0
ly = 100.0  # Definition of the size of the domain
subsH = 10.0  # Height of the substrate
domainSize = (lx**2 + ly**2) ** 0.5

# Generation of the mesh
mesh = generate_refined_plate(lx, ly, subsH, mesh_size_fine=1, mesh_size_coarse=1)

# Determining the number of dofs
dofs_per_node = 2
n_dofs = dofs_per_node * mesh.coords.shape[0]
n_eles = mesh.elements.shape[0]
coords = mesh.coords
elements = mesh.elements

# Defining the material properties
Emat = 210000  # young's modulus of the material
Emin = Emat * 1e-9  # young'd modulus of the void region
nu = 0.3

op = Operator(mesh, element.Tri3())  # Tatva operator is defined
n_quad = op.element.quad_points.shape[0]  # number of quadrature points per element

# calculation of the centroid of the triangle
centroids = (
    mesh.coords[mesh.elements[:, 0]]
    + mesh.coords[mesh.elements[:, 1]]
    + mesh.coords[mesh.elements[:, 2]]
) / 3
centroidsX = centroids[:, 0]
centroidsY = centroids[:, 1]

# substrate elements / nodes
subsEleIndex = jnp.where(centroidsY < 0)[0]
subsNodeIndex = jnp.where(mesh.coords[:, 1] < 0)[0]

# print(subsNodeIndex)
# print(jnp.max(subsNodeIndex))
# print(jnp.min(subsNodeIndex))
# breakpoint()

# domain elements / nodes
domEleIndex = jnp.where(centroidsY > 0)[0]
domNodeIndex = jnp.where(mesh.coords[:, 1] > 0)[0]

subsDomainSep = jnp.zeros(n_eles)
subsDomainSep = subsDomainSep.at[domEleIndex].set(0)
subsDomainSep = subsDomainSep.at[subsEleIndex].set(1)

plot_mesh(mesh, subsDomainSep)
# plt.show()
# breakpoint()

meshDomain = Mesh(
    coords=mesh.coords[domNodeIndex, :2], elements=mesh.elements[domEleIndex, :3]
)

# Design variable
xTopo = jnp.ones(n_eles)

# Substrate Filter
subsFilter = jnp.ones(n_eles)
subsFilter = subsFilter.at[subsEleIndex].set(0)

# Definition of the number of subparts and orientation variable
numSubParts = 1
theta = [90]
bead = 3
maxLayers = jnp.ceil((domainSize / bead)).astype(jnp.int32)
layerArray = jnp.arange(-maxLayers, maxLayers + 1, 1)
maxNumLayers = layerArray.size
PI = 1.0 - (jnp.arange(numSubParts, -1, -1).reshape(-1, 1) / numSubParts)
PI = PI.at[-1, 0].add(0.1)
tPhys = jnp.zeros(n_eles)
tPhys = tPhys.at[domEleIndex].set(0.5)
tPhys = tPhys.at[subsEleIndex].set(0.5)

# Segmentation generation: element below a time stamp
eBelowPI = jnp.zeros((n_eles, 1, numSubParts + 1))
paraSubsPart = 100

for i in range(1, numSubParts + 1):
    eBelowPI = eBelowPI.at[:, 0, i].set(
        1.0 - (1.0 / (1.0 + jnp.exp(-paraSubsPart * (tPhys - PI[i, 0]))))
    )

# Elements between the time stamps
eInPI = jnp.zeros((n_eles, numSubParts))

for i in range(1, numSubParts + 1):
    eInPI = eInPI.at[:, i - 1].set(
        jnp.ravel(xTopo) * (eBelowPI[:, 0, i] - eBelowPI[:, 0, i - 1])
    )

totalEInPI = jnp.sum(eInPI, axis=0)

# Planar layer generation: defining the distance field
distPhi = jnp.zeros((n_eles, numSubParts))

for i in range(1, numSubParts + 1):
    sind = math.sin(math.radians(theta[i - 1]))
    cosd = math.cos(math.radians(theta[i - 1]))
    distPhi = distPhi.at[:, i - 1].set((sind * centroidsY) + (cosd * centroidsX))


# Layer generation
@jax.jit
def compute_new_layer(eInPI, distPhi, layerNum, paraH2, shapeH2, bead):
    distPhi_i = distPhi[:, None, :]  # (N, 1, numPI)
    layer_j = layerNum[None, :, 0]  # (1, maxLayerUpdated)
    layer_j_i = layer_j[:, :, None]  # (1, maxLayerUpdated, 1)
    delta = distPhi_i - (bead / 2.0) - bead * (layer_j_i - 1)
    inner = 1.0 - (shapeH2 / (bead**2)) * (delta**2)
    sigmoid = 1.0 / (1.0 + jnp.exp(paraH2 * inner))
    eInPI_exp = eInPI[:, None, :]  # (N, j, i)
    newLayer = eInPI_exp * (1.0 - sigmoid)
    return newLayer


paraLayerGen = 100
shapeLayerGen = 4
layerNum = layerArray.reshape(-1, 1)

newLayer = compute_new_layer(
    eInPI, distPhi, layerNum, paraLayerGen, shapeLayerGen, bead
)


@jax.jit
def compute_depo_struct(newLayer):
    layer_cumsum = jnp.cumsum(newLayer, axis=1)
    end_vals = layer_cumsum[:, -1, :]  # (totalEles, numPI)
    offsets = jnp.concatenate(
        [
            jnp.zeros((newLayer.shape[0], 1), dtype=newLayer.dtype),
            jnp.cumsum(end_vals[:, :-1], axis=1),
        ],
        axis=1,
    )
    depoStruct = layer_cumsum + offsets[:, None, :]
    return depoStruct


depoStruct = compute_depo_struct(newLayer)
depoStruct = depoStruct.at[subsEleIndex, :, :].set(1)

plot_mesh(mesh, depoStruct[:, 50, 1])
plt.show()


# @jax.jit
def compute_load_vector(
    newLayer, i, j, xTopo, subsFilter, elements, coords, Emat, Emin, nu
):
    # Heat flux load

    alphaExpCoeff = 9e-6
    Tmelt = 1500
    Tinter = 150
    deltaTemp_ihs = Tmelt - Tinter
    ihsStrain_xx_ihs = -alphaExpCoeff * deltaTemp_ihs
    ihsStrain_yy_ihs = -alphaExpCoeff * deltaTemp_ihs

    strain_ihs = jnp.array([ihsStrain_xx_ihs, ihsStrain_yy_ihs, 0.0]).reshape(-1, 1)

    L = jnp.sqrt(4.0 / jnp.sqrt(3.0))

    # Natural derivatives
    dN = jnp.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    # Jacobian
    J = jnp.array([[L, 0.0], [L / 2, jnp.sqrt(3) * L / 2]])

    # Global derivatives
    dN_xy = jnp.linalg.inv(J) @ dN

    Nx = dN_xy[0, :]
    Ny = dN_xy[1, :]

    B_ihs = jnp.array(
        [
            [Nx[0], 0, Nx[1], 0, Nx[2], 0],
            [0, Ny[0], 0, Ny[1], 0, Ny[2]],
            [Ny[0], Nx[0], Ny[1], Nx[1], Ny[2], Nx[2]],
        ]
    )

    D_ihs = ((Emat - Emin) / (1 - nu**2)) * jnp.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1 - nu)]]
    )

    stress_ihs = D_ihs @ strain_ihs
    load_ihs = B_ihs.T @ stress_ihs

    print(strain_ihs)
    print(strain_ihs.shape)
    print(stress_ihs)
    print(stress_ihs.shape)
    print(B_ihs)
    print(B_ihs.shape)
    print(load_ihs)
    print(load_ihs.shape)

    # strain

    # qLoad = (1 / 3) * (65 * 1.6) * (math.sqrt(3) / 4) * jnp.array([1.0, 1.0, 1.0])
    penalQ = 3

    coordxy1 = coords[elements[:, 0]]
    coordxy2 = coords[elements[:, 1]]
    coordxy3 = coords[elements[:, 2]]

    coordx1 = coordxy1[:, 0]
    coordy1 = coordxy1[:, 1]
    coordx2 = coordxy2[:, 0]
    coordy2 = coordxy2[:, 1]
    coordx3 = coordxy3[:, 0]
    coordy3 = coordxy3[:, 1]

    j11 = coordx2 - coordx1
    j12 = coordy2 - coordy1
    j21 = coordx3 - coordx1
    j22 = coordy3 - coordy1

    jacobian = jnp.abs(j11 * j22 - j12 * j21)
    Connectivity = jnp.array(
        [
            [2 * elements[:, 0]],
            [2 * elements[:, 0] + 1],
            [2 * elements[:, 1]],
            [2 * elements[:, 1] + 1],
            [2 * elements[:, 2]],
            [2 * elements[:, 2] + 1],
        ]
    )

    dummyConnectivity = Connectivity.T.flatten()
    # print(elements)
    # print(elements.flatten())
    # print(Connectivity)
    # print(Connectivity.T)
    # print(dummyConnectivity)
    # print(jnp.min(dummyConnectivity.flatten()))

    # print(elements.shape)
    # print(dummyConnectivity.shape)
    #
    row_idx = jnp.asarray(dummyConnectivity, dtype=jnp.int64)

    dummyNewLayer = jacobian * xTopo * subsFilter * newLayer[:, j, i]

    dummyNewLayerKron = jnp.kron(
        jnp.ravel(dummyNewLayer) ** penalQ, jnp.ones((load_ihs.shape[0],))
    ).reshape(-1, 1)
    dummyLoad = jnp.repeat(load_ihs, n_eles, axis=0)

    vals = dummyNewLayerKron * dummyLoad
    dummyLoad = jnp.zeros((n_dofs, 1))

    dummyLoad = dummyLoad.at[row_idx].add(vals)

    dummyLoad1 = dummyLoad[2 * elements][:, 0]
    dummyLoad2 = dummyLoad[2 * elements + 1][:, 0]
    dummyLoad3 = dummyLoad[2 * elements][:, 1]
    dummyLoad4 = dummyLoad[2 * elements + 1][:, 1]
    dummyLoad5 = dummyLoad[2 * elements][:, 2]
    dummyLoad6 = dummyLoad[2 * elements + 1][:, 2]

    return (
        dummyLoad,
        dummyLoad1,
        dummyLoad2,
        dummyLoad3,
        dummyLoad4,
        dummyLoad5,
        dummyLoad6,
    )


# computing the penality on the layer stiffness matrix
@jax.jit
def compute_Epenal(xTopo, depoStruct, i_idx, j_idx):
    penalE = 3
    dummyDepostruct = depoStruct[:, j_idx, i_idx]
    return Emin + ((xTopo * dummyDepostruct) ** penalE) * (Emat - Emin)


# defining the stress function
@autovmap(eps=2, mu=0, lmbda=0)
def compute_stress(eps, mu, lmbda):
    return 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)


# defining strain function
@autovmap(grad_u=2)
def compute_strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)


# Calculating strain energy density
@autovmap(grad_u=2, mu=0, lmbda=0, Epenal=0)
def strain_energy_density(grad_u, mu, lmbda, Epenal):
    eps = compute_strain(grad_u)
    sigma = compute_stress(eps, mu, lmbda)
    return 0.5 * Epenal * jnp.einsum("ij,ij->", sigma, eps)


@jax.jit
def total_energy_full(u_flat: Array, Epenal) -> Array:
    """Compute the total energy of the system."""
    u = u_flat.reshape(-1, 2)
    u_grad = op.grad(u)
    e_density = strain_energy_density(u_grad, mat.mu, mat.lmbda, Epenal)
    return op.integrate(e_density)


class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float
    lmbda: float

    @classmethod
    def from_youngs_poisson_2d(
        cls, E: float, nu: float, plane_stress: bool = False
    ) -> "Material":
        mu = E / 2 / (1 + nu)
        if plane_stress:
            lmbda = 2 * nu * mu / (1 - nu)
        else:
            lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
        return cls(mu=mu, lmbda=lmbda)


mat = Material.from_youngs_poisson_2d(1, 0.3, plane_stress=True)

# Dirichlet boundary conditions at the substrate nodes (held fixed)
fixed_dofs = jnp.concatenate(
    [subsNodeIndex * dofs_per_node, (subsNodeIndex * dofs_per_node) + 1]
)
free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), fixed_dofs)

# print(fixed_dofs.shape)
# print(subsNodeIndex.shape)
# print(free_dofs.shape)
# print(n_dofs)
# breakpoint()

# fixed_Dofs = subsNodeIndex

# ---------------------------------------------------------------------------
# Sparsity pattern (one scalar DOF per node).
# (A Compound state could own this layout, but a single field doesn't need it here.)
# Compressed sparse row
# ---------------------------------------------------------------------------
csr_full = sparse.pattern_from_mesh(mesh, n_dofs_per_node=2)

# ---------------------------------------------------------------------------
# Lifter: Dirichlet BC holding the substrate at u = v = 0. We solve for the *total*
# displacement on the free DOFs; lift_from_zeros places u=v=0 on the fixed DOFs.
# ---------------------------------------------------------------------------
lifter = Lifter(n_dofs, Fixed(fixed_dofs, float(0)))

# Reduced sparsity (free DOFs only) + graph coloring for sparse assembly
reduced_sparsity_pattern = lifter.reduce_sparsity(csr_full)
colored_matrix = sparse.ColoredMatrix.from_csr(reduced_sparsity_pattern)


# breakpoint()
# ---------------------------------------------------------------------------
# Reduced backward-Euler potential on free DOFs + sparse-differentiation assembly
# ---------------------------------------------------------------------------
@jax.jit
def total_energy(u_free: Array, Epenal) -> Array:
    """Compute the total energy of the system."""
    u_full = lifter.lift_from_zeros(
        u_free
    )  # jnp.zeros(n_dofs).at[free_dofs].set(u_free.ravel())
    return total_energy_full(u_full, Epenal)


# Reduced residual R(T_free) (gradient of the potential w.r.t. the free DOFs).
residual_full = jax.grad(total_energy_full)
residual = jax.grad(total_energy)


@jax.jit
def fn(u_free: Array, fext: Array, Epenal: Array) -> Array:
    res = residual(u_free, Epenal)
    return f_ext - res


hessian_fn = sparse.jacfwd(fn, colored_matrix)


# print(f_ext_full.shape)
# print(f1.shape)
# print(f2.shape)
# print(f3.shape)
# print(f4.shape)
# print(f5.shape)
# print(f6.shape)
# print(n_dofs)
# print(n_eles)

# plot_force_vectors(mesh, f_ext_full)


import time
from functools import partial


def newton_direct_solver(
    u,
    gradient,
    jacobian,
    tol: float = 1e-8,
    max_iter: int = 10,
):
    """Newton's method with a sparse LU factorization of the tangent.

    ``gradient(u)`` is the residual  f_ext - dE/du,  so its Jacobian J = -K with K
    the (positive-definite) stiffness. The Newton update solves  J du = -residual,
    i.e.  K du = residual, which is what is factorized below.

    The energy is quadratic in u, so this converges in a single step; the loop is
    kept so the residual is actually measured rather than assumed. Unlike CG, the
    accuracy here does not degrade with cond(K) -- which is ~1e9 because of the
    Emin floor on the not-yet-deposited elements.
    """
    residual = np.asarray(gradient(u))
    norm_res = float(np.linalg.norm(residual))
    norm_res_0 = norm_res

    for iiter in range(max_iter):
        if norm_res <= tol * max(norm_res_0, 1.0):
            break

        prev_norm_res = norm_res

        start_time = time.perf_counter()
        J = jacobian(u).to_csr()  # d(residual)/du = -K
        assemble_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        lu = splu((-J).tocsc())  # factorize K
        du = lu.solve(residual)  # K du = residual
        solve_time = time.perf_counter() - start_time

        u = u + jnp.asarray(du)

        start_time = time.perf_counter()
        residual = np.asarray(gradient(u))
        residual_time = time.perf_counter() - start_time
        norm_res = float(np.linalg.norm(residual))

        print(
            f"  iter {iiter + 1}: |R| = {norm_res:.3e}"
            f"  |R|/|R0| = {norm_res / max(norm_res_0, 1e-300):.3e}"
            f"  (assemble {assemble_time:.2f} s,"
            f" LU solve {solve_time:.2f} s,"
            f" residual {residual_time:.2f} s)"
        )

        # The energy is quadratic, so one LU solve is already exact: any further
        # iteration only churns at the roundoff floor. Stop as soon as the residual
        # stops dropping by a clear factor.
        if norm_res >= 0.1 * prev_norm_res:
            break

    print(
        f"  Residual: {norm_res:.2e} (relative {norm_res / max(norm_res_0, 1e-300):.2e})"
    )

    return u, norm_res


u_init = jnp.zeros(n_dofs)
u_history = [u_init]

for i in range(1, numSubParts + 1):
    for j in range(60, maxNumLayers + 1):
        u_curr = u_init  # last converged full temperature
        u_free = lifter.reduce(u_curr)  # free DOFs of the total temperature

        Epenal = jnp.broadcast_to(
            compute_Epenal(xTopo, depoStruct, i - 1, j - 1).flatten()[:, None],
            (n_eles, n_quad),
        )

        f_ext_full, f1, f2, f3, f4, f5, f6 = compute_load_vector(
            newLayer, i - 1, j - 1, xTopo, subsFilter, elements, coords, Emat, Emin, nu
        )
        f_ext_full = f_ext_full.flatten()
        f_ext = f_ext_full.at[free_dofs].get()

        partial_fn = jax.jit(partial(fn, fext=f_ext, Epenal=Epenal))

        # Sparse tangent via graph-coloring forward-mode AD: n_colors JVPs give the
        # full matrix, which is then LU-factorized.
        partial_jacobian = jax.jit(partial(hessian_fn, fext=f_ext, Epenal=Epenal))

        u_sol, norm_res = newton_direct_solver(
            u=u_free,
            gradient=partial_fn,
            jacobian=partial_jacobian,
        )

        print(u_sol.shape)
        print(n_dofs)
        u_curr = u_curr.at[free_dofs].set(u_sol.ravel())

        print(u_curr)
        defMesh = deformMesh(mesh, u_curr)
        print(Epenal / jnp.max(Epenal))

        plot_mesh(defMesh, (Epenal.ravel()) / jnp.max(Epenal))
        plt.show()
        print(jnp.max(u_sol))
        print(jnp.max(f_ext))
        print(jnp.max(f_ext_full))

        # breakpoint()

        u_history.append(u_sol)
        u_init = u_curr  # Update the initial guess for the next iteration

# plot_mesh(mesh,f_extv/(jnp.abs(f_extv)+1e-6))
# breakpoint()
