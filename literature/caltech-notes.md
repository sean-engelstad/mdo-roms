# Notes on Elizabeth Qian's 'Caltech Notes'

## POD (Proper Orthogonal Decomposition)

Take a snapshot matrix $S = [s_1, ..., s_m]$ and perform SVD $S = U \Sigma V^T$ with rank-revealing SVD giving the reduced basis matrix $U_r$ of left eigenvectors. The matrix $U$ in fact minimizes the reconstruction error $\min_{U} || (I - UU^T) S ||_2 $, while $U_r$ minimizes the rank $r$ reconstruction error.

## Galerkin POD
Galerkin POD on a linear static system gives the following:
$$ K u = F, \quad u = U_r \hat{u} \rightarrow U_r^T K U_r \hat{u} = U_r^T f $$  $$ \hat{K} \hat{u} = \hat{f} \quad \hat{K} = U_r^T K U_r $$
Note other methods like Petrov-Galerkin POD just minimizes reconstruction norm in a different space than Galerkin. Galerkin is the easiest to compute and as the same convergence properties as other methods, so is often the best one to use.

## Linear, Affine Parameter Case
With an affine parameter case for parameter $x$,  $K(x) = \sum_{i=1}^N \phi(x_i) K_i$, we can pre-compute the reduced stiffness matrices for each parameter $\hat{K}_i = U_r^T \hat{K} U_r$ offline, and have fast online assembly fo the stiffness matrix $\hat{K} = \sum_{i=1}^N \phi(x_i) \hat{K}_i $. 

## Nonlinear, Hyper-Reduction
With Galerkin POD, split the linear and nonlinear parts of the state into separate terms.
$$ K_L u + R_{NL}(u) = f $$
Applying Galerkin POD with rank $r$ reduced basis $U_r$ we have the following reduced residual:
$$ U_r^T K_L U_r \hat{u} + U_r^T R_{NL}(U_r \hat{u}) = U_r^T f $$  $$ \rightarrow \hat{K}_L + U_r^T R_{NL}(U_r \hat{u}) = \hat{f}$$  Here the linear term with its affine dependence can be pre-computed and saved offline for fast online computation of $\hat{K}_L$. While the nonlinear reduced residual term $U_r^T R_{NL}(U_r \hat{u})$ scales by the full-problem in the general case. However, on GPUs we can still compute this nonlinear state cheaply and may not require hyper-reduction. We can just use the reduced basis to reduce solve time and symbolic factorization. The methods to reduce the computational cost of the nonlinear reduced residual term are DEIM and QDEIM, methods which select a subset of residual evaluations to approximate full residual.

### DEIM method
Let $f_i = R_{NL}(u_i)$ be nonlinear residual evaluations saved and computed at the same time as the regular POD snapshots $F = [f_i]$ with an SVD performed $F = \Phi \Xi \Psi^T$ with $\Phi_{\rho}$ the $\rho$ reduced-rank basis for the residuals. This will allow us to compute the eigenmodes of residual evaluation, assuming the residual itself has some level of low-rank or low frequency behavior (which is generally true although we will need to compute residual at least once in each TACS component). 

We then approximate the DEIM residual by a linear combination of the residual eigenvectors basis:
$$ R_{NL,DEIM}(\hat{u}) = \Phi_{\rho} \cdot c(\hat{u})$$   The reduced nonlinear residual modal coefficients $c(\hat{u})$ are found through by a least-squares regression which enforces the residual evaluations to match at our prescribed subset of points. That is let $P$ be a selection matrix (all ones and zeros which selects which subset of points to evaluate) $P = [e_{p_1}, ..., e_{p_{\rho}}]$. Then we enforce at each of the $\rho$ chosen points:
$$ P^T \Phi_{\rho} c(\hat{u}) = P^T R_{NL}(U_r \hat{u})$$  The matrix $P^T \Phi_{\rho}$ is actually a $\rho \times \rho$ square matrix which we invert to find $c(\hat{u})$:
$$ c(\hat{u}) = (P^T \Phi_{\rho})^{-1} P^T R_{NL}(U_r \hat{u}) $$  The final approximate nonlinear residual is then:
$$ R_{NL,DEIM}(\hat{u}) = \Phi_{\rho} (P^T \Phi_{\rho})^{-1} P^T R_{NL}(U_r \hat{u}) $$  If you also don't want to form the full state, you can use locality properties of the true residual $R_{NL}$, for example in finite element problems we don't need to form the full state anymore, we just need:
$$ P^T R_{NL}(U_r \hat{u}) = \sum_{e \in E} P_e^T K_e(U_{r,e} \hat{u}) \cdot U_{r,e} \hat{u} $$   Where $u_e = U_{r,e} \hat{u}$ is the part of the full state in these subset of elemeents which are needed by the reduced residual. Also $P_e^T$ extracts only certian nodes out of the full element residual. Are there other strategies which still use more of the nodes of that single element? A reduced mesh that is. See DEIM for finite elements. But nevertheless, this reduces the full-state computation and the reduced residual evaluation at the same time. Computing full-state is pretty cheap though and we would probably do that in parallel on the GPU or localize it.

* see p. 41 for a greedy strategy to compute the locations for the reduced residual, aka the $P$ matrix nz entries.

### QDEIM method
The QDEIM method uses the QR algorithm to find the $P$ matrix which leads to a more efficient DEIM strategy.
