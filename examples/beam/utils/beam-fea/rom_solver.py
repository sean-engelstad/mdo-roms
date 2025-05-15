import numpy as np
import scipy as sp
from solver import *

# inspired mostly from this paper from Farhat; don't need to use hyper-ROM since this is a linear solve first
# [Design optimization using hyper-reduced-order models]
#   (https://link.springer.com/article/10.1007/s00158-014-1183-y?sa_campaign=email%2Fevent%2FarticleAuthor%2FonlineFirst)

class BeamFemRom(BeamFem):
    def __init__(self, nxe:int, nxh:int, E:float, b:float, L:float, rho:float, 
                 qmag:float, ys:float, rho_KS:float, dense:bool=False):
        
        # call superclass constructor
        super(BeamFemRom,self).__init__(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense)

        # important ROM data (for n full dof, m red dof)
        self.V = None # n x m ROB matrix
        self.m = None
        self.Khat_arr = None # num_design x m x m

        self.Khat = None # reduced mat (m x m)
        self.fhat = None # reduced forces (m x 1)
        self.uhat = None # reduced disp (m x 1)
        self.psis_hat = None # reduced stress adjoint (m x 1)

    @classmethod
    def from_HDM(cls, beam_fem:BeamFem):
        return cls(
            nxe=beam_fem.nxe,
            nxh=beam_fem.nxh,
            E=beam_fem.E,
            b=beam_fem.b,
            L=beam_fem.L,
            rho=beam_fem.rho,
            qmag=beam_fem.qmag,
            ys=beam_fem.ys,
            rho_KS=beam_fem.rho_KS,
            dense=beam_fem._dense,
        )

    # now add ROM methods here
    # HDM methods still inherited
    def gen_param_list(self, lb, ub, hvec0, num_designs:int):
        hvec_list = [hvec0]
        rand_vals = lb + np.random.rand(num_designs-1, hvec0.shape[0]) * (ub - lb)
        
        for idesign in range(num_designs-1):
            hvec_list += [rand_vals[idesign,:]]
        return hvec_list

    def offline_ROM_training(self, hvec_list:list, rank:int):
        # construct snapshots of solved displacements at each design
        # run linear solve at a bunch of random designs to get snapshot matrix W
        W = []
        for idesign, hvec in enumerate(hvec_list):
            # print(f"solving {idesign=}")
            super(BeamFemRom,self).solve_forward(hvec)
            unorm = self.u / np.linalg.norm(self.u)
            W += [np.expand_dims(unorm, axis=-1)]
        W = np.concatenate(W, axis=1)
        # print(f"{W.shape=}")
        # print(f"{W=}")

        # now do SVD on W snapshot matrix
        U, s, VH = np.linalg.svd(W)
        Ur = U[:,:rank]

        pod_rel_err = np.linalg.norm(s[(rank+1):]) / np.linalg.norm(s)
        print(f"{pod_rel_err=}")
        # TODO : later could choose error norm eps and select rank 
        # based on error tol

        # could use s to get error estimates (see Farhat paper)
        self.V = Ur # this is our POD reduced order basis now
        self.m = rank

        self._compute_Khat_fhat_offline()

    def _compute_Khat_fhat_offline(self):
        """compute the th_i(x_i) * Khat_i => store list of Khat_e for each design variable"""

        self.Khat_arr = np.zeros((self.nxe, self.m, self.m))

        E = self.E; b = self.b # all but the hvec[ielem]**3 DV part here
        Ke_nom = E * b / 12.0 * get_kelem(self.xscale)
        felem_nom = get_felem(self.xscale)

        xvec = [(ielem+0.5) * self.dx for ielem in range(self.num_elements)]
        qvec = [self.qmag * np.sin(4.0 * np.pi * xval / self.L) for xval in xvec]
        force = np.zeros((self.num_dof))

        for ielem in range(self.nxe): # in this case one design var per element now
            # this still scales by num dof only because # design vars does,
            # in harder problems will use TACS / GPU_FEM and TACS components with not nearly as many DV as elements
            # so this will be much better reduction for linear problems

            # for element-based case
            # V^T G_e^T K_e G_e V = V_e^T K_e V_e (where V_e are parts of V in that element so Ve is num_dof_elem x m matrix)
            local_dof_conn = np.array(self.dof_conn[ielem])
            Ve = self.V[local_dof_conn,:]
            Khat_i = Ve.T @ Ke_nom @ Ve

            idesign = ielem # one design var per element here
            self.Khat_arr[idesign,:,:] = Khat_i

            # add global forces
            q = qvec[ielem]
            local_conn = np.array(self.dof_conn[ielem])
            np.add.at(force, local_conn, q * felem_nom)

        # apply BCs to global forces (just in case, prob not needed since ROB has these BCs)
        bcs = [0, 2 * (self.num_nodes-1)]
        for bc in bcs:
            force[bc] = 0.0
        force = np.expand_dims(force, axis=-1)

        # compute reduced forces
        self.fhat = self.V.T @ force
        # print(f"{self.fhat=}")
        return
    
    def _compute_red_mat_vec(self, hvec):
        # dense matrix now by # POD modes
        helem_vec = self.get_helem_vec(hvec)
        self.Khat = np.einsum('ijk,i->jk', self.Khat_arr, helem_vec**3)
        return
    
    def _compute_dRdx(self, hvec):
        # print(f"called rom solver dRdx")
        Rgrad = [0.0] * self.num_elements
        for ielem in range(self.num_elements):
            dKhat_dx = self.Khat_arr[ielem,:,:] * 3 * hvec[ielem]**2
            # print(f"{self.psis_hat.shape=} {dKhat_dx.shape=} {self.uhat.shape=}")
            Rgrad[ielem] = float(self.psis_hat.T @ dKhat_dx @ self.uhat)
        # print(f"{Rgrad=}")
        # return np.array(Rgrad)
        return np.array(self.helem_to_hred_vec(Rgrad))
    
    def solve_forward(self, hred):
        helem_vec = self.get_helem_vec(hred)
        # solve reduced disp coeffs for linear POD ROB
        self._compute_red_mat_vec(helem_vec)
        # self.Khat is dense matrix so dense linear solve
        self.uhat = np.linalg.solve(self.Khat, self.fhat)
        return self.uhat
    
    def solve_adjoint(self, hred):
        helem_vec = self.get_helem_vec(hred)
        # solve reduced psi coeffs for linear POD ROB
        self._compute_red_mat_vec(helem_vec)
        KT = self.Khat.T
        dstress_du = self._compute_dstressdu(helem_vec)
        stress_rhs = self.V.T @ dstress_du # dstress/du => dstress/duhat conversion
        stress_rhs = np.expand_dims(stress_rhs, axis=-1)

        # KT is dense matrix so dense linear solve
        self.psis_hat = np.linalg.solve(KT, -stress_rhs)
        return self.psis_hat
    
    def get_full_disp(self):
        self.u = self.V @ self.uhat
        return self.u
    
    def get_functions(self, hvec):
        self.get_full_disp()
        return super(BeamFemRom,self).get_functions(hvec)
    
    def get_function_gradients(self, hvec):
        self.get_full_disp()
        return super().get_function_gradients(hvec)

    def plot_pod_modes(self):
        colors = plt.cm.jet(np.linspace(0.0, 1.0, self.m))
        plt.figure()
        for imode in range(self.m):
            plt.plot(self.xvec, self.V[0::2, imode], color=colors[imode])
        plt.xlabel("x")
        plt.ylabel("POD Unit Modal Disp")
        plt.show()

    def plot_stresses(self, hred):
        helem_vec = self.get_helem_vec(hred)
        colors = plt.cm.jet(np.linspace(0.0, 1.0, self.m))
        plt.figure()

        # first solve HDM
        super(BeamFemRom,self).solve_forward(hred)
        HDM_stresses = super(BeamFemRom,self)._compute_stresses(helem_vec)
        plt.plot(self.xvec[1:], HDM_stresses, label="HDM")

        # then solve ROM
        self.solve_forward(hred)
        self.get_full_disp()
        ROM_stresses = self._compute_stresses(helem_vec)
        plt.plot(self.xvec[1:], ROM_stresses, '--', label="ROM")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Stress(x)")
        plt.show()

    def plot_disp_compare(self, hvec):
        colors = plt.cm.jet(np.linspace(0.0, 1.0, self.m))
        plt.figure()

        # first solve HDM
        super(BeamFemRom,self).solve_forward(hvec)
        plt.plot(self.xvec, self.u[0::2], label="HDM")

        # then solve ROM
        self.solve_forward(hvec)
        self.get_full_disp()
        plt.plot(self.xvec, self.u[0::2], '--', label="ROM")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Disp(x)")
        plt.show()

    def plot_pod_disps(self, num_plot=10):
        num_plot = min([num_plot, self.m])
        colors = plt.cm.jet(np.linspace(0.0, 1.0, self.m))
        plt.figure()
        for imode in range(num_plot):
            plt.plot(self.xvec, self.uhat[imode] * self.V[0::2, imode], 
                     color=colors[imode])
        plt.xlabel("x")
        plt.ylabel("POD Solved Modal Disps")
        plt.show()

