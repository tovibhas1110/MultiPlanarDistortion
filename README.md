# AM4Micro: Designing Material Microstructures Through Additive Manufacturing Sequence Planning

## Overview

This repository contains a Python-based finite element framework for **predicting mechanical distortion in large-scale metal additive manufacturing**, particularly for components manufactured using multiple build directions.

In large-scale metal additive manufacturing, a component can be segmented into multiple sub-parts, each manufactured using a distinct build direction. This approach offers greater manufacturing flexibility but introduces challenges associated with thermal gradients, residual stresses, and mechanical distortion.

During additive manufacturing, localized heating and cooling generate high temperature gradients and rapid cooling rates, resulting in manufacturing-induced deformation.

To address this challenge, the project implements a **finite element-based distortion prediction framework using the inherent strain method**. Rather than explicitly simulating the complete transient thermal history, the inherent strain approach provides a computationally efficient approximation of the accumulated effects of the manufacturing process.

The implementation builds upon the [Tatva computational framework](https://tatva.ch/), developed by **Dr. Mohit Pundir at ETH Zürich**, and utilizes JAX for efficient numerical computations and automatic differentiation.

## Research Objectives

The main objectives are:

- Predict mechanical distortion in large-scale metal additive manufacturing.
- Investigate the influence of different build directions and sub-part configurations.
- Approximate manufacturing-induced thermal effects using the inherent strain method.
- Develop a computationally efficient finite element implementation.
- Establish a foundation for distortion-aware additive manufacturing sequence planning and optimization.

## Methodology

### 1. Multi-directional Additive Manufacturing

The component is segmented into multiple sub-parts, which may be manufactured using different build directions.

This enables the evaluation of manufacturing strategies beyond conventional single-direction deposition.

### 2. Conformal Finite Element Meshing

The framework supports conformal meshing for discretizing the component geometry into finite elements.

Conformal discretization ensures consistent mesh connectivity across adjacent regions and sub-part interfaces.

### 3. Inherent Strain-Based Distortion Prediction

The accumulated effects of the additive manufacturing process are approximated using the inherent strain method.

The governing mechanical equilibrium equation is:

$$
\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}
$$

The constitutive relationship is:

$$
\boldsymbol{\sigma} =
\mathbf{C} :
\left(
\boldsymbol{\varepsilon} -
\boldsymbol{\varepsilon}^{*}
\right)
$$

where:

- $\boldsymbol{\sigma}$ — Stress tensor
- $\mathbf{C}$ — Elastic constitutive tensor
- $\boldsymbol{\varepsilon}$ — Total strain tensor
- $\boldsymbol{\varepsilon}^{*}$ — Prescribed inherent strain tensor

After finite element discretization, the governing system becomes:

$$
\mathbf{K}\mathbf{u} = \mathbf{f}_{\mathrm{inh}}
$$

where:

- $\mathbf{K}$ — Global stiffness matrix
- $\mathbf{u}$ — Nodal displacement vector
- $\mathbf{f}_{\mathrm{inh}}$ — Equivalent nodal force vector associated with inherent strains

Solving the resulting linear system provides the displacement field used to estimate manufacturing-induced distortion.

### 4. Automatic Differentiation and JAX

The implementation utilizes the **JAX library** to enable:

- Automatic differentiation of numerical operations.
- Efficient stiffness matrix calculations.
- Just-in-time compilation of supported computational routines.
- A foundation for sensitivity analysis and gradient-based optimization.

## Key Features

- **Conformal Meshing:** Finite element discretization of complex geometries.
- **Finite Element Analysis:** Numerical prediction of mechanical distortion.
- **Inherent Strain Method:** Computationally efficient approximation of manufacturing-induced deformation.
- **Automatic Differentiation:** JAX-enabled differentiation of numerical operations.
- **Efficient Computation:** Integration of JAX, NumPy, and SciPy.
- **Multi-directional Manufacturing:** Investigation of different build directions and sub-part configurations.

## Installation and Requirements

### Prerequisites

- Python 3 (compatible with the installed JAX and Tatva versions)
- NumPy
- SciPy
- JAX
- Matplotlib
- Gmsh
- Meshio
- Tatva

### Getting Started

**Step 1: Clone the repository**

```bash
git clone https://github.com/tovibhas1110/MultiPlanarDistortion.git
```

**Step 2: Navigate to the repository**

```bash
cd MultiPlanarDistortion
```

**Step 3: Create a virtual environment (recommended)**

```bash
python3 -m venv env
source env/bin/activate
```

**Step 4: Install the Python dependencies**

```bash
python -m pip install numpy scipy jax matplotlib gmsh meshio
```

Install the Tatva framework separately following its official documentation:

https://tatva.ch/

**Step 5: Run the Python implementation**

```bash
python MPDis_Obracket.py
```

**Note:** Compatible Python, JAX, and Tatva versions are required. Additional configuration may be necessary depending on the operating system.

## Research Contributions

The main technical contributions include:

1. Development of a finite element-based framework for estimating manufacturing-induced distortion.
2. Incorporation of the inherent strain method to approximate the mechanical effects of additive manufacturing.
3. Support for conformal finite element meshing and multi-directional deposition configurations.
4. Integration of JAX-based numerical computation and automatic differentiation.
5. Development of a computational foundation for future optimization of manufacturing segmentation, build directions, and deposition sequences.

## Applications

The framework is relevant to:

- Wire and Arc Additive Manufacturing (WAAM)
- Directed Energy Deposition (DED)
- Large-scale metal additive manufacturing
- Multi-directional deposition planning
- Distortion-aware design optimization
- Additive manufacturing sequence planning

## Research Publication

For details on the distortion prediction methodology, refer to:

**Mishra, V., Wu, J. (2026). Differentiable modelling and optimization of multi-planar slicing for multi-axis additive manufacturing **

*Structural and Multidisciplinary Optimization.*

DOI: https://doi.org/10.1007/s00158-025-04240-3

## Computational Framework

This implementation builds upon the **Tatva framework**, developed by Dr. Mohit Pundir at ETH Zürich, Switzerland.

Website: https://tatva.ch/

## Acknowledgements

This research is associated with the **AM4Micro project** at Delft University of Technology, The Netherlands.

The implementation builds upon the Tatva computational framework developed at ETH Zürich.

## Author

**Dr. ir. Vibhas Mishra**  
Delft University of Technology  
Delft, The Netherlands

GitHub: https://github.com/tovibhas1110

