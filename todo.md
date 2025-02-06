# TODO List for "GPUs for Efficient ROMs of Geometric Nonlinear Structural Optimization"

## Literature


## Demos
- [ ] investigate beam case, see if numerical issues in SVD
    * should have smooth stresses apparently even if POD just on disps
    * check # modes, # snapshots for disp, 1st and 2nd derivatives of disp
* plate structural optimizations:
    - [ ] baseline axial and transverse plate geometric nonlinear optimization (from Reddy et al.)
    - [ ] hyper-reduction example
    - [ ] GPUs to do this in GPU_FEM (compare to python results here)
    - [ ] compare with operator learning strategies