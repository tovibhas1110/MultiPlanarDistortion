# Project Name

AM4Micro: Designing Material Microstructures Through Additive Manufacturing Sequence Planning 

## Overview

In a large-scale metal additive manufacturing process, the part is segmented into multiple sub-parts, and each sub-part is realized 
using a distinct build direction. High temperature gradients and cooling rates induce mechanical distortion in the part. To 
estimate the part distortion, we present a finite element-based code to estimate the part distortion. The thermal effects are considered using
the inherent strain methods. 

To get more details on the method to estimate distortion, 
refer to the article: https://doi.org/10.1007/s00158-025-04240-3 
by Dr. ir. Vibhas Mishra at Delft University of Technology, Delft, The Netherlands.

The code is based on the Tatva framework: https://tatva.ch/
developed by Dr. Mohit Pundir at ETH Zurich, Zurich, Switzerland.

## Features

- Conformal meshing
- Automatic Differentiation-enabled Stiffness Matrix Calculation
- JAX library usage for fast computation

### Requirements

- Python >= 3.x
- NumPy
- SciPy
- JAX
- Matplotlib

### Installation

Clone the repository:
```bash
git clone https://github.com/tovibhas1110/MultiPlanarDistortion.git
cd MultiPlanarDistortion
