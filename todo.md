# TODO List for "GPUs for Efficient ROMs of Geometric Nonlinear Structural Optimization"

## Literature


## Demos
- [ ] investigate beam case with high stresses, see if numerical issues in SVD
    * should have smooth stresses apparently even if POD just on disps
    * check # modes, # snapshots for disp, 1st and 2nd derivatives of disp
* plate structural optimizations:
    - [ ] baseline axial and transverse plate geometric nonlinear optimization (from Reddy et al.)
    - [ ] hyper-reduction example
    - [ ] GPUs to do this in GPU_FEM (compare to python results here)
    - [ ] compare with operator learning strategies
- [ ] put MELD on the GPU
- [ ] put Arrow FVM from Brian on the GPU (my own repo)
- [ ] demo fluids ROMs with Arrow (w/ GPU for reduced stiffness)
- [ ] demo aeroelastic ROM! (by proposal time), this would be sick