"""
POD (proper orthogonal decomposition) class

Takes in an array of snapshots and any metadata
"""

import numpy as np
import matplotlib.pyplot as plt

class POD:
    def __init__(self, U, **kwargs):
        """
        Args: 
            U (numpy array): array of snapshots
        """

        self.U = U
        self.num_snapshots = np.shape(U)[0]

        # unpack metadata
        if 'metadata' in kwargs:
            for key, value in kwargs['metadata'].items():
                setattr(self, key, value)
            self.metadata = True

    def compute_pod(self):
        self._compute_covariance()
        self._eigen_decomp()
        self._compute_phi()

    def _compute_covariance(self):
        """
        Computes the snapshot covariance matrix based on the matrix U
        """
        self.C_s = 1/(self.num_snapshots -1) * np.matmul(self.U,self.U.T)

    def _eigen_decomp(self):
        """
        Computes the eigenvalues and eigenvectors of snapshot covariance matrix C_s
        """
        lam_s, A_s = np.linalg.eig(self.C_s)

        self.lam_s = lam_s
        self.A_s = A_s
        
    def _compute_phi(self):
        """
        Compute the POD modes phi (still in matrix form)
        """

        self.phi = np.matmul(self.U.T, self.A_s)

    
    def compute_modes(self):
        """
        Computes the POD modes from phi by restructuring the data into an array of shape (nvar, num_shapshots, nx, ny, nz)
        
        nvar, nx, ny, and nz need to be present in the metadata 
        """

        if len(self._check_metadata(['nvar', 'nx', 'ny', 'nz']))>0:
            print('Missing the relevant metadata.')
            return
        else:
            self.modes = self.phi.T.reshape(-1, self.nvar, self.nx, self.ny, self.nz)

    def reconstruct(self, num_modes):
        """
        Reconstruct U with the first 'num_modes' of modes 
        """

        return np.matmul(self.A_s[:,:num_modes], self.phi[:,:num_modes].T)

    def reconstruct_double(self, num_modes, comp1, comp2):
        """
        Reconstruct U comp1 U comp2 with the first 'num_modes' of modes 
        """

        if len(self._check_metadata(['nvar', 'nx', 'ny', 'nz']))>0:
            print('Missing the relevant metadata.')
            return

        total = self.nx * self.ny * self.nz
        
        return np.sum(self.lam_s[:num_modes] * self.phi[comp1*total:(comp1+1)*total,:num_modes] * self.phi[comp2*total:(comp2+1)*total,:num_modes], axis=1)

    def reconstruct_triple(self, num_modes, comp1, comp2, comp3):
        """
        Reconstruct U comp1 U comp2 with the first 'num_modes' of modes 
        """

        if len(self._check_metadata(['nvar', 'nx', 'ny', 'nz']))>0:
            print('Missing the relevant metadata.')
            return

        total = self.nx * self.ny * self.nz
        
        return np.sum(self.lam_s[:num_modes] * self.phi[comp1*total:(comp1+1)*total,:num_modes] * self.phi[comp2*total:(comp2+1)*total,:num_modes], axis=1)


    def _check_metadata(self, vars):
        """
        Check if variable exists in POD object

        Args:
        vars (list): list of variable names to check
        """

        missing = []

        for var in vars:
            if not hasattr(self, var):
                missing.append(var)

        return missing

    def plot_slice(self, mode, component, x=None, y=None, z=None, **kwargs):
        """
        Plot slice of pod mode
        """

        if len(self._check_metadata([ 'x', 'y', 'z']))>0:
            print('Missing the relevant metadata.')
            return
        else:
            fig, ax = plt.subplots()
            if x is not None:
                xloc = np.argmin(np.abs(self.x - x))

                im = ax.imshow(self.modes[mode, component, xloc, ...].T, origin='lower', 
                               extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)], **kwargs)
                
                ax.set_xlabel('y/D')
                ax.set_ylabel('z/D')

                fig.colorbar(im)
                return fig, ax
            elif y is not None:
                yloc = np.argmin(np.abs(self.y - y))

                im = ax.imshow(self.modes[mode, component, :, yloc, :].T, origin='lower', 
                               extent=[np.min(self.x), np.max(self.x), np.min(self.z), np.max(self.z)], **kwargs)
                
                ax.set_xlabel('x/D')
                ax.set_ylabel('z/D')

                fig.colorbar(im)
                return fig, ax
            elif z is not None:
                zloc = np.argmin(np.abs(self.z - z))

                im = ax.imshow(self.modes[mode, component, ..., zloc].T, origin='lower', 
                               extent=[np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y)], **kwargs)
                
                ax.set_xlabel('x/D')
                ax.set_ylabel('y/D')

                fig.colorbar(im)
                return fig, ax
            else:
                print("Need to specify x, y or z location.")
                return




