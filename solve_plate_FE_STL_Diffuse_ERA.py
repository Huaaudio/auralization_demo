import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import time
import sys

def main(freq_in=None, f_res=None, m_ratio=None, eta_res=None, filename='TestPlateForJiahua.mat', fig_flag = True):
    
    # Input: regular CQUAD4 shell FE model of a finite plate

    ###########################################################################
    ## Load full plate FE model
    ###########################################################################
    # Note: Ensure this filename matches what you generated in the Setup script
    # e.g., 'TestPlateForJiahua.mat' or 'Plate_model_Alu_2mm_40x30cm_8x6UCs.mat'
    mat_filename = filename
    try:
        # loadmat with squeeze_me=True helps remove singleton dimensions
        # struct_as_record=False allows accessing fields as attributes (mat.K instead of mat['K'])
        mat_data = sio.loadmat(mat_filename, squeeze_me=True, struct_as_record=False)
    except FileNotFoundError:
        print(f"Error: Could not find {mat_filename}. Please check the filename or run the Setup script.")
        return

    # Extract variables
    # We must handle the object structure from the .mat file
    model_matrices = mat_data['model_matrices']
    
    # Extract matrices (ensure they are sparse CSC/CSR for efficiency)
    K_raw = model_matrices.K
    M_raw = model_matrices.M
    C_raw = model_matrices.C
    
    # Handle K4gg (might be missing in some saves, but present in setup)
    if hasattr(model_matrices, 'K4gg'):
        K4gg_raw = model_matrices.K4gg
    else:
        K4gg_raw = sp.csc_matrix(K_raw.shape)

    GRID = mat_data['GRID']
    
    # Scalar parameters
    L_UCx = float(mat_data['L_UCx'])
    L_UCy = float(mat_data['L_UCy'])
    N_UCx = int(mat_data['N_UCx'])
    N_UCy = int(mat_data['N_UCy'])
    N_elem_UCx = int(mat_data['N_elem_UCx'])
    N_elem_UCy = int(mat_data['N_elem_UCy'])
    Lx = float(mat_data['Lx'])
    Ly = float(mat_data['Ly'])
    E = float(mat_data['E'])
    nu = float(mat_data['nu'])
    t = float(mat_data['t'])
    
    # Grid Data Processing for Python
    # GRID.ID is 1-based from MATLAB. Convert to 0-based for Python indices.
    GRID_ID = GRID.ID.astype(int)
    GRID_Boundary = GRID.Boundary.astype(int)
    GRID_Interior = GRID.Interior.astype(int)
    GRID_X1 = GRID.X1
    GRID_X2 = GRID.X2
    GRID_X3 = GRID.X3
    
    N_nodesy = int(mat_data['N_nodesy'])
    N_nodesx = int(mat_data['N_nodesx'])

    # For analytical check
    h = 0.002 # m
    rho = 2700 # kg/m^3

    # Symmetrize
    # Note: transposing complex sparse matrix in scipy conjugates by default. 
    # MATLAB's .' is transpose without conjugate. .T in scipy is transpose.
    # We assume K, M, C are real from the setup, K4gg might be used for damping.
    
    K = (K_raw + K_raw.T)/2 + 1j*(K4gg_raw + K4gg_raw.T)/2
    M = (M_raw + M_raw.T)/2
    C = (C_raw + C_raw.T)/2

    # BCs
    BC_type = 'clamped' # 'supported' or 'clamped', 'free' doesn't make sense for STL

    # Scatterers
    scatterer_type = 'resonator' # 'none' or 'resonator' or 'pointmass'
    # Arrangement
    arrangement = 'ordered' # 'ordered' (nominal), 'disordered' or 'random'
    N_elem_dis_x = 0 # in case of 'disordered', indicate max distance from nominal
    N_elem_dis_y = N_elem_dis_x

    # For point mass and resonator (constant scatterer properties)
    m_ratio = 0.5 if m_ratio is None else m_ratio  # Scatterer mass ratio vs beam (m_scatterer/m_UC) [-]
    # For resonator (constant scatterer properties)
    f_res = 500 if f_res is None else f_res # Tuned resonator frequency of the resonators [Hz]
    f_ratio = [] # Overrides f_res
    ksi_res = 0 # Viscous damping ratio of the resonators [-]
    eta_res = 0.005 if eta_res is None else eta_res # Structural damping coefficient of the resonators [-]

    ## Analysis parameters
    # Frequency range
    if freq_in is not None:
        freq = freq_in
    else:
        freq = np.arange(20, 3005, 5) # 20:5:3000 (include end)
    omega = 2 * np.pi * freq

    # Modal basis
    n_modes = 105 # 105,63,153

    # Blocked pressure field excitation
    # Angles
    theta = np.arange(0, 78 + 78/5, 78/5) * np.pi/180
    psi = np.arange(0, 90 + 90/2, 90/2) * np.pi/180
    
    n_theta = len(theta)
    n_psi = len(psi)
    
    # Meshgrid for angles
    # MATLAB default meshgrid is 2D xy.
    thetas, psis = np.meshgrid(theta, psi) 
    
    # Flattening (column-major order 'F' to match MATLAB (:))
    angles = np.column_stack((thetas.flatten(order='F'), psis.flatten(order='F')))
    
    diffuse_tf = 1

    rho0 = 1.225 # Density of air [kg/m^3]
    c0 = 340 # Sound speed [m/s]
    k0_vec = omega/c0 # Wave number vector
    
    # These will be matrices (N_freq x N_angles) or similar
    # We compute kx0 and ky0 inside the loop or broadcasting
    P0 = 1 # (complex) amplitude of the incident wave [Pa]

    # Radiated sound calculation using ERA based Rayleigh integral of velocity field
    deltaL = 0.025 # Subsampling resolution for ERA grid

    # Out-of-plane dofs
    # Python 0-based: ID is 1-based, so (ID-1) is the index.
    # DOF 3 (Z) corresponds to index 2.
    # Formula: 6 * (ID - 1) + 2
    dof_plate_uz = 6 * (GRID_ID - 1) + 2

    # Frequencies for full field post-processing
    freq_plot = [1630]
    plot_v_tf = 1

    # Set solution approach
    ModalMOR_tf = 1 # Modal approach (if 0, FOM is calculated)

    ###########################################################################
    ## Create forcing vector
    ###########################################################################
    # Matrix of force vectors (in columns) per angle per frequency:
    # Blocked pressure field is frequency-dependent

    print('Setting up forces...')
    t_start = time.time()
    
    # F dimensions: (Total DOFs, N_freqs, N_angles)
    F = np.zeros((K.shape[0], len(freq), len(angles)), dtype=complex)
    dof_F = 6 * (GRID_ID - 1) + 2
    
    # Precompute k0 for all freqs
    k0 = omega / c0
    
    for i in range(len(angles)):
        theta_i = angles[i, 0]
        psi_i = angles[i, 1]
        
        # Calculate trace wavenumbers (Frequency dependent)
        # kx0(i,:) in Matlab implies kx0 is likely (N_angles x N_freq) or similar depending on k0
        # k0 is vector of length N_freq. 
        kx0_i = k0 * np.sin(theta_i) * np.cos(psi_i)
        ky0_i = k0 * np.sin(theta_i) * np.sin(psi_i)
        
        # GRID.X1 and X2 are vectors of length N_nodes.
        # We need outer product or broadcasting.
        # X1 is (N_nodes, ), kx0_i is (N_freq, )
        # Result should be (N_nodes, N_freq)
        
        # phase = kx*x + ky*y
        phase = np.outer(GRID_X1, kx0_i) + np.outer(GRID_X2, ky0_i)
        
        # f_b = 2 * P0 * exp(...) * Area_factor
        area_factor = (L_UCx * L_UCy) / ((N_elem_UCx) * (N_elem_UCy))
        f_b = 2 * P0 * np.exp(-1j * phase) * area_factor
        
        # Assign to F
        # F[dof, freq, angle]
        # f_b is (N_nodes, N_freq). We map N_nodes to their DOFs.
        F[dof_F, :, i] = f_b

    print(f"Elapsed: {time.time() - t_start:.4f}s")

    ###########################################################################
    ## Add scatterers (append, in case of TVAs)
    ###########################################################################
    
    # Helper to calculate indices logic from Matlab
    # Matlab: idx_centerY = N_elem_UCy/2+(N_elem_UCx/2)*N_nodesy+1+[0:N_UCy-1]*N_elem_UCy;
    # Convert to 0-based indexing carefully.
    
    # Base ranges
    range_UCy = np.arange(N_UCy)
    range_UCx = np.arange(N_UCx)
    
    # Logic matching Matlab exactly (using floats then casting)
    # Note: Matlab 1-based index '1' becomes 0-based offset.
    # Term: N_elem_UCy/2 + (N_elem_UCx/2)*N_nodesy
    # This assumes N_elem_UCx is even.
    
    term1 = N_elem_UCy / 2
    term2 = (N_elem_UCx / 2) * N_nodesy
    
    if arrangement == 'ordered':
        # idx_centerY
        # Matlab: ... + 1 + [0:N_UCy-1]*N_elem_UCy
        # Python: ... + range * N_elem_UCy (indices are 0-based, so we don't add the '1' for the base)
        # BUT GRID.ID in Matlab is 1-based. Let's calculate the 1-based ID first then convert.
        
        # 1-based calculation
        idx_centerY_1based = term1 + term2 + 1 + range_UCy * N_elem_UCy
        
        # idx_center
        # Matlab: idx_centerY.'+[0:N_UCx-1]*N_elem_UCx*N_nodesy
        idx_center_1based = idx_centerY_1based.reshape(-1, 1) + range_UCx * N_elem_UCx * N_nodesy
        
        ScatLoc = idx_center_1based.flatten(order='F').astype(int)
        
    elif arrangement == 'disordered':
        idx_centerY_1based = term1 + term2 + 1 + range_UCy * N_elem_UCy
        idx_center_1based = idx_centerY_1based.reshape(-1, 1) + range_UCx * N_elem_UCx * N_nodesy
        
        if (N_elem_dis_x < N_elem_UCx/2) and (N_elem_dis_y < N_elem_UCy/2):
            # datasample in Matlab: samples with replacement.
            dx_pool = np.arange(-N_elem_dis_x, N_elem_dis_x + 1)
            dy_pool = np.arange(-N_elem_dis_y, N_elem_dis_y + 1)
            
            delta_x = np.random.choice(dx_pool, N_UCy * N_UCx)
            delta_y = np.random.choice(dy_pool, N_UCy * N_UCx)
            
            idx_center_flat = idx_center_1based.flatten(order='F')
            
            idx_delta = idx_center_flat + delta_x * N_nodesy + delta_y
            ScatLoc = idx_delta.astype(int)
        else:
            raise ValueError('n_elem_dis_x < N_elem_UCx/2 and n_elem_dis_y < N_elem_UCy/2 required!')
            
    elif arrangement == 'random':
        # Matlab: idx_rand = randperm(numel(GRID.Interior), N_UCx*N_UCy)
        # ScatLoc = GRID.Interior(idx_rand)
        idx_rand = np.random.permutation(len(GRID_Interior))[:N_UCx*N_UCy]
        ScatLoc = GRID_Interior[idx_rand] # These are already 1-based IDs
    else:
        raise ValueError('Enter correct argument for arrangement!')

    # Visualize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Reshape for mesh plotting
    # Matlab: X = reshape(GRID.X1,[N_nodesy,N_nodesx]);
    X_mesh = GRID_X1.reshape((N_nodesx, N_nodesy)).T # Numpy reshape is row-major, Matlab is col-major
    Y_mesh = GRID_X2.reshape((N_nodesx, N_nodesy)).T
    Z_mesh = np.zeros_like(X_mesh)
    
    ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color='blue', linewidth=0.5)
    
    # Boundary nodes
    # Convert 1-based ID to 0-based index
    bound_idx = GRID_Boundary - 1
    ax.scatter(GRID_X1[bound_idx], GRID_X2[bound_idx], GRID_X3[bound_idx], c='g', marker='+')
    
    # Scatterers
    scat_idx = ScatLoc - 1
    if scatterer_type in ['resonator', 'pointmass']:
        ax.scatter(GRID_X1[scat_idx], GRID_X2[scat_idx], GRID_X3[scat_idx], c='k', marker='*')
        
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.view_init(elev=90, azim=0) # Top view
    # plt.axis('equal') # Matplotlib 3d axis equal is tricky, skipping for now
    
    # Determine scatterer base dofs
    # 0-based indices for DOFs
    # 1 -> 0, 2 -> 1, 3 -> 2
    dof_base_ux = 6 * (ScatLoc - 1) + 0
    dof_base_uy = 6 * (ScatLoc - 1) + 1
    dof_base_uz = 6 * (ScatLoc - 1) + 2
    
    # Derived scatterer properties
    N_scatterers = N_UCx * N_UCy
    m_UC = rho * L_UCx * L_UCy * t
    m_discrete = m_ratio * m_UC
    
    # For resonator
    if isinstance(f_ratio, (list, np.ndarray)) and len(np.atleast_1d(f_ratio)) > 0:
        D_bend = E * t**3 / (12 * (1 - nu**2))
        f_Bragg = (D_bend * (2*np.pi)**2 / (rho * t * (2*L_UCx)**4))**(1/2)
        f_res = f_ratio * f_Bragg
        print(f'Sub-wavelength limit = {f_Bragg} Hz')
        print(f'f_res = {f_res} Hz')
        
    k_discrete = (f_res * 2 * np.pi)**2 * m_discrete * (1 + 1j * eta_res)
    
    if eta_res > 0 and ksi_res > 0:
        raise ValueError('Eta_res and ksi_res cannot be simultaneously non-zero!')
        
    c_discrete = ksi_res * 2 * np.sqrt(m_discrete * k_discrete)
    
    # Update Matrices
    if scatterer_type == 'none':
        pass
    
    elif scatterer_type == 'resonator':
        print('Adding scatterers to system of equations...')
        t_start = time.time()
        
        current_size = K.shape[0]
        # Determine resonator mass uz dofs
        # New DOFs are appended at the end
        dof_res_mass_uz = np.arange(current_size, current_size + N_scatterers)
        
        # Expand Matrices
        # We need to add N_scatterers rows and cols.
        # Construct sparse blocks
        new_size = current_size + N_scatterers
        
        # We will use lil_matrix for easier modification or construct COO/CSR directly.
        # Given the "update" logic in the loop, let's build the update triplets
        
        rows_K = []
        cols_K = []
        data_K = []
        
        rows_M = []
        cols_M = []
        data_M = []
        
        rows_C = []
        cols_C = []
        data_C = []
        
        # Local matrices
        M_res_val = m_discrete
        # K_res matrix: [k, -k; -k, k]
        # C_res matrix: [c, -c; -c, c]
        # Dofs involved: [dof_base_uz, dof_res_mass_uz]
        
        for i in range(N_scatterers):
            db = dof_base_uz[i]
            dr = dof_res_mass_uz[i]
            dofs = [db, dr]
            
            # K update
            # K_res = [[k, -k], [-k, k]]
            vals_k = [[k_discrete, -k_discrete], [-k_discrete, k_discrete]]
            for r in range(2):
                for c in range(2):
                    rows_K.append(dofs[r])
                    cols_K.append(dofs[c])
                    data_K.append(vals_k[r][c])
                    
            # M update
            # M_res = [[0, 0], [0, m]]
            vals_m = [[0, 0], [0, M_res_val]]
            for r in range(2):
                for c in range(2):
                    if vals_m[r][c] != 0:
                        rows_M.append(dofs[r])
                        cols_M.append(dofs[c])
                        data_M.append(vals_m[r][c])
                        
            # C update
            vals_c = [[c_discrete, -c_discrete], [-c_discrete, c_discrete]]
            for r in range(2):
                for c in range(2):
                     if vals_c[r][c] != 0: # optimization for zero
                        rows_C.append(dofs[r])
                        cols_C.append(dofs[c])
                        data_C.append(vals_c[r][c])
        
        # Resize original matrices to new size (padding with zeros)
        # sparse.resize works but can be slow. 
        # Better to creating new matrix with bmat or coo.
        
        def resize_sparse(A, new_shape):
            # A is (N, N), want (M, M)
            # Create (M-N, M) and (N, M-N) blocks? No, simpler to just re-create from COO
            # Or use Resize method of LIL/CSR
            A_csr = A.tocsr()
            A_csr.resize(new_shape)
            return A_csr

        K = resize_sparse(K, (new_size, new_size))
        M = resize_sparse(M, (new_size, new_size))
        C = resize_sparse(C, (new_size, new_size))
        
        # Pad F
        F_padding = np.zeros((N_scatterers, len(freq), len(angles)), dtype=complex)
        F = np.concatenate((F, F_padding), axis=0)
        
        # Apply updates
        K = K + sp.coo_matrix((data_K, (rows_K, cols_K)), shape=(new_size, new_size))
        M = M + sp.coo_matrix((data_M, (rows_M, cols_M)), shape=(new_size, new_size))
        C = C + sp.coo_matrix((data_C, (rows_C, cols_C)), shape=(new_size, new_size))
        
        print(f"Elapsed: {time.time() - t_start:.4f}s")
        
    elif scatterer_type == 'pointmass':
        print('Adding scatterers to system of equations...')
        t_start = time.time()
        
        rows_M = []
        cols_M = []
        data_M = []
        
        for i in range(N_scatterers):
            dofs = [dof_base_ux[i], dof_base_uy[i], dof_base_uz[i]]
            for d in dofs:
                rows_M.append(d)
                cols_M.append(d)
                data_M.append(m_discrete)
                
        M = M + sp.coo_matrix((data_M, (rows_M, cols_M)), shape=M.shape)
        print(f"Elapsed: {time.time() - t_start:.4f}s")
        
    else:
        raise ValueError('Enter correct argument for scatterer_type!')

    ###########################################################################
    ## Apply BCs
    ###########################################################################
    # Determine DOFs to delete
    # Matlab: (GRID.Boundary-1)*6 + 1 ...
    
    dofs_delete = []
    
    if BC_type == 'supported':
        print('Fixed translations BCs...')
        for node_id_1b in GRID_Boundary:
            # 1-based node ID -> 0-based node idx -> 0-based DOF idx
            node_idx = node_id_1b - 1
            # DOFs 1, 2, 3 (ux, uy, uz) -> indices 0, 1, 2
            dofs_delete.extend([node_idx*6 + 0, node_idx*6 + 1, node_idx*6 + 2])
            
    elif BC_type == 'clamped':
        print('Clamped BCs...')
        for node_id_1b in GRID_Boundary:
            node_idx = node_id_1b - 1
            # DOFs 1-6 -> indices 0-5
            dofs_delete.extend([node_idx*6 + k for k in range(6)])
            
    elif BC_type == 'free':
        print('Free BCs...')
        dofs_delete = []
        
    dofs_delete = np.unique(dofs_delete).astype(int)
    dofs_all = np.arange(K.shape[0])
    dofs_keep = np.setdiff1d(dofs_all, dofs_delete)
    
    # Reduce Matrices (Slicing)
    K = K[dofs_keep, :][:, dofs_keep]
    M = M[dofs_keep, :][:, dofs_keep]
    C = C[dofs_keep, :][:, dofs_keep]
    F = F[dofs_keep, :, :]
    
    ###########################################################################
    ## SOLUTION
    ###########################################################################
    
    d_red = None
    V = None
    
    if ModalMOR_tf:
        ## Modal MOR
        print('Calculating eigenmodes ...')
        t_start = time.time()
        # We must cast to CSC explicitly for SuperLU (used by eigsh).
        K_solve = K.real.tocsc()
        M_solve = M.real.tocsc()
        # eigs(real(K), real(M), n_modes, 0)
        # Python eigsh with sigma=0 performs shift-invert to find eigenvalues near 0
        vals, vecs = spla.eigsh(K_solve, M=M_solve, k=n_modes, sigma=0, which='LM')
        
        # Sort output (eigsh doesn't guarantee order)
        idx_sort = np.argsort(vals)
        vals = vals[idx_sort]
        V = vecs[:, idx_sort]
        
        eigfreqs = (np.sqrt(vals)) / (2 * np.pi)
        f_max = eigfreqs[-1]
        print(f"Elapsed: {time.time() - t_start:.4f}s")
        
        ## Projection
        print('Projecting system matrices ...')
        t_start = time.time()
        
        M_red = V.T @ M @ V
        K_red = V.T @ K @ V
        C_red = V.T @ C @ V
        
        # F_red = V.' * F
        # F is (Ndof, Nfreq, Nangle). V is (Ndof, Nmodes).
        # We need tensor contraction.
        # F_red(mode, freq, angle) = sum_dof (V(dof, mode) * F(dof, freq, angle))
        # Equivalent to V.T @ F_reshaped
        
        F_shape = F.shape
        F_flat = F.reshape(F_shape[0], -1) # (Ndof, Nfreq*Nangle)
        F_red_flat = V.T @ F_flat
        F_red = F_red_flat.reshape(n_modes, F_shape[1], F_shape[2])
        
        print(f"Elapsed: {time.time() - t_start:.4f}s")
        
        ## Direct FRF calculation
        # Matrices are now dense and small
        d_red = np.zeros((n_modes, len(freq), len(angles)), dtype=complex)
        
        print('Solving system of equations (MOR)...')
        t_start = time.time()
        
        # Pre-convert sparse to dense if they aren't already (projection usually makes them dense-ish or we treat as dense)
        # Given n_modes ~100, dense solve is fastest.
        if sp.issparse(K_red): K_red = K_red.toarray()
        if sp.issparse(M_red): M_red = M_red.toarray()
        if sp.issparse(C_red): C_red = C_red.toarray()
        
        for i in range(len(freq)):
            w = omega[i]
            Z_red = K_red + 1j * w * C_red - w**2 * M_red
            
            # RHS for this freq: all angles
            rhs = F_red[:, i, :] # (N_modes, N_angles)
            
            # Solve
            sol = la.solve(Z_red, rhs)
            d_red[:, i, :] = sol
            
        print(f"Elapsed: {time.time() - t_start:.4f}s")

    else:
        ## FOM
        d = np.zeros((K.shape[0], len(freq), len(angles)), dtype=complex)
        print('Solving system of equations (FOM)...')
        t_start = time.time()
        
        for i in range(len(freq)):
            w = omega[i]
            Z = K + 1j * w * C - w**2 * M
            
            rhs = F[:, i, :]
            
            # Sparse solve
            sol = spla.spsolve(Z, rhs)
            d[:, i, :] = sol
            
        print(f"Elapsed: {time.time() - t_start:.4f}s")
        
    ## Average out-of-plane plate FRF
    # Reconstruct full vector
    # u_all dimensions: (Total original DOFs, N_freq, N_angles)
    u_all = np.zeros((len(dofs_all), len(freq), len(angles)), dtype=complex)
    
    if ModalMOR_tf:
        print('Back-projection...')
        t_start = time.time()
        # u = V * d_red
        # Tensor logic: u(dof, freq, angle) = sum_mode (V(dof, mode) * d_red(mode, freq, angle))
        
        # Loop over angles to manage memory or use reshaping
        d_red_flat = d_red.reshape(n_modes, -1)
        u_keep_flat = V @ d_red_flat
        u_keep = u_keep_flat.reshape(len(dofs_keep), len(freq), len(angles))
        
        u_all[dofs_keep, :, :] = u_keep
        print(f"Elapsed: {time.time() - t_start:.4f}s")
    else:
        u_all[dofs_keep, :, :] = d
        
    # Extract uz
    u_plate_uz = u_all[dof_plate_uz, :, :]
    
    # Velocity
    v_plate_uz = np.zeros_like(u_plate_uz)
    for i in range(len(angles)):
        # Broadcasting omega (N_freq,) over (N_nodes, N_freq)
        v_plate_uz[:, :, i] = u_plate_uz[:, :, i] * omega[None, :] * 1j # NOTE: v = i*omega*u. Matlab implicitly handled complex arithmetic? Script says u*omega, but usually v = j*w*u. Wait, script uses: v... = u... * omega. I will stick to script logic, but usually it implies complex. The script computes RMS later.
        # Check Matlab script: v_plate_uz(:,:,i) = u_plate_uz(:,:,i).*omega; 
        # It misses the '1i'. However, for RMS/Power, the phase shift '1i' doesn't matter, only magnitude.
        
    if plot_v_tf:
        # RMS over nodes? Script: rms(v_plate_uz(:,1:end,:)) -> RMS along first dimension (nodes)
        v_plate_uz_RMS = np.sqrt(np.mean(np.abs(v_plate_uz)**2, axis=0)) # (N_freq, N_angles)
        
        plt.figure()
        for i in range(len(angles)):
            plt.semilogy(freq, v_plate_uz_RMS[:, i])
        plt.title('RMS FRF')
        label = f'Modal, n = {n_modes}' if ModalMOR_tf else 'FOM'
        plt.legend([label])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('RMS(v_z) [m/s]') # Velocity unit

    ## Visualize displacement field response
    
    # Prepare grid for plotting (Standard Meshgrid for plotting)
    X_plot = GRID_X1.reshape((N_nodesx, N_nodesy)).T
    Y_plot = GRID_X2.reshape((N_nodesx, N_nodesy)).T
    
    for f_target in freq_plot:
        idx_f = np.argmin(np.abs(freq - f_target))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Reshape Re(u_z)
        # u_plate_uz is (N_nodes, N_freq, N_angles). take angle 0
        uz_plot = u_plate_uz[:, idx_f, 0].real.reshape((N_nodesx, N_nodesy)).T
        
        ax.plot_surface(X_plot, Y_plot, uz_plot, cmap='viridis', edgecolor='none')
        ax.set_title(f'Modal, n = {n_modes}, freq = {freq[idx_f]} Hz')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('Re(u_z) [m]')
    
    # Subsampled velocity field plots
    # Define subsampled grid axes
    # Matlab: deltaL/2:deltaL:Lx-deltaL/2
    x_sub = np.arange(deltaL/2, Lx, deltaL) # check upper bound logic
    y_sub = np.arange(deltaL/2, Ly, deltaL)
    X_sub, Y_sub = np.meshgrid(x_sub, y_sub) # Default xy
    
    # Create Interpolator
    # We need to construct the axes of the original data.
    x_orig = np.linspace(0, Lx, N_nodesx)
    y_orig = np.linspace(0, Ly, N_nodesy)
    
    for f_target in freq_plot:
        idx_f = np.argmin(np.abs(freq - f_target))
        
        # Original Data (reshaped to 2D grid)
        # Note: RegularGridInterpolator expects (points_x, points_y) and data(x, y).
        # Our X_plot, Y_plot are (N_nodesy, N_nodesx).
        # We need data in (y, x) if using y_orig, x_orig order, or transpose.
        
        # Let's align with Cartesian.
        # Z_data[j, i] corresponds to y[j], x[i]
        
        # u_plate is flattened F-order (column major). 
        # reshaped to (N_nodesy, N_nodesx) makes rows=y, cols=x.
        vz_data = v_plate_uz[:, idx_f, 0].real.reshape((N_nodesy, N_nodesx), order='F') 
        
        # Interpolator
        interp = RegularGridInterpolator((y_orig, x_orig), vz_data, method='linear')
        
        # Points to interpolate
        # grid of (Y_sub, X_sub)
        pts = np.column_stack((Y_sub.ravel(), X_sub.ravel()))
        vz_sub_flat = interp(pts)
        vz_sub = vz_sub_flat.reshape(X_sub.shape)
        
        # Plot Original
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_plot, Y_plot, vz_data, cmap='viridis')
        ax.set_title(f'Original Re(vz), f={freq[idx_f]} Hz')
        
        # Plot Subsampled
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_sub, Y_sub, vz_sub, cmap='viridis')
        ax.set_title(f'Subsampled Re(vz), f={freq[idx_f]} Hz')

    ###########################################################################
    ## SOUND TRANSMISSION LOSS CALCULATION (Rayleigh integral)
    ###########################################################################
    ## Incident powers
    # W_i = Lx*Ly*P0^2*cos(thetas(:))/(2*rho0*c0);
    # angles[:,0] is theta flattened
    W_i = Lx * Ly * P0**2 * np.cos(angles[:, 0]) / (2 * rho0 * c0)

    ## Via ERA approach
    print('Calculating transmitted sound power')
    t_start = time.time()
    
    # Elementary radiator centers (already defined X_sub, Y_sub)
    n_subsampled = X_sub.size
    
    # Distances
    # X_sub.ravel() is vector.
    # We need matrix (N, N) of distances.
    # dist[i, j] = sqrt( (xi - xj)^2 + (yi - yj)^2 )
    coords = np.column_stack((X_sub.ravel(), Y_sub.ravel()))
    from scipy.spatial.distance import cdist
    distance = cdist(coords, coords)
    
    # Radiation Impedance & Power
    W_t_ERA = np.zeros((len(angles), len(freq)))
    S = deltaL * deltaL
    a = (S / np.pi)**0.5
    
    # Diagonal indices for replacement
    # In numpy we can use np.fill_diagonal
    
    for c_freq in range(len(freq)):
        k_c = k0[c_freq]
        w_c = omega[c_freq]
        
        # Z_rad calculation
        # exp(-1i * k * r) / r
        # Warning: divide by zero on diagonal, handle temporary inf
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.exp(-1j * k_c * distance) / distance
            
        Z_rad = (rho0 * 1j * w_c * S * term) / (2 * np.pi)
        
        # Fix diagonal
        # val = rho0 * c0 * (1 - exp(-1i * k * a))
        diag_val = rho0 * c0 * (1 - np.exp(-1j * k_c * a))
        np.fill_diagonal(Z_rad, diag_val)
        
        # Velocity interpolation for all angles at this freq
        # v_plate_uz shape: (N_nodes, N_freq, N_angles)
        # Extract for this freq: (N_nodes, N_angles)
        v_data_freq = v_plate_uz[:, c_freq, :]
        
        # We need to interpolate this column-by-column (angle-by-angle)
        # Or reshape to (N_nodesy, N_nodesx, N_angles) and interpolate
        
        # Optimization: Create interpolator once per freq is not needed, grid is static.
        # But data changes.
        # Let's loop angles.
        
        for i_ang in range(len(angles)):
            # Reshape data to grid
            v_grid = v_data_freq[:, i_ang].reshape((N_nodesy, N_nodesx), order='F')
            
            # Interpolate
            interp = RegularGridInterpolator((y_orig, x_orig), v_grid, method='linear')
            v_sub_flat = interp(pts) # (n_subsampled, )
            
            # Transmitted power
            # W = real(v' * Z * v) * S / 2
            # v is complex
            val = np.vdot(v_sub_flat, Z_rad @ v_sub_flat) # vdot handles conjugate of first arg
            W_t_ERA[i_ang, c_freq] = np.real(val) * S / 2
            
    # Transmission coefficients
    # W_t_ERA is (N_angles, N_freq)
    # W_i is (N_angles, )
    tau_ERA = W_t_ERA / W_i[:, None]
    
    # STL
    # Avoid log(0)
    STL_ERA = 10 * np.log10(1.0 / (tau_ERA + 1e-20))
    
    print(f"Elapsed: {time.time() - t_start:.4f}s")

    # Diffuse integration
    STL_ERA_d = None
    
    if diffuse_tf:
        print('Diffuse integration')
        t_start = time.time()
        
        # Reshape tau to (N_theta, N_psi, N_freq) for integration
        # angles was formed by flatten(order='F') of meshgrid(theta, psi)
        # theta changes fast, psi changes slow
        # tau_ERA corresponds to angles.
        
        # Thetas is (N_psi, N_theta) in meshgrid default? No.
        # Meshgrid: theta (N_theta), psi (N_psi).
        # thetas (N_psi, N_theta), psis (N_psi, N_theta)
        # flatten('F'): [theta[0], psi[0]], [theta[1], psi[0]]...
        # So first dimension corresponds to theta iteration.
        
        tau_reshaped = tau_ERA.reshape((len(theta), len(psi), len(freq)), order='F')
        
        # Integration weights
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        weight_theta = sin_theta * cos_theta
        
        int_over_psi_teller = np.zeros(len(freq))
        int_over_psi_noemer = np.zeros(len(freq))
        
        if n_psi > 1:
            for cfreq in range(len(freq)):
                # Integrate over theta for each psi
                # tau_reshaped[:, cpsi, cfreq] is vector of thetas
                
                int_over_theta_teller = np.zeros(n_psi)
                int_over_theta_noemer = np.zeros(n_psi)
                
                for cpsi in range(n_psi):
                    tau_vals = tau_reshaped[:, cpsi, cfreq]
                    int_over_theta_teller[cpsi] = np.trapz(tau_vals * weight_theta, theta)
                    int_over_theta_noemer[cpsi] = np.trapz(weight_theta, theta)
                    
                # Integrate over psi
                int_over_psi_teller[cfreq] = np.trapz(int_over_theta_teller, psi)
                int_over_psi_noemer[cfreq] = np.trapz(int_over_theta_noemer, psi)
                
            tau_ERA_d = int_over_psi_teller / int_over_psi_noemer
            
        else:
            # Axis symmetric or single psi
             for cfreq in range(len(freq)):
                 # tau_ERA is (N_theta, freq) effectively
                 tau_vals = tau_ERA[:, cfreq]
                 num = np.trapz(tau_vals * weight_theta, theta)
                 den = np.trapz(weight_theta, theta)
                 tau_ERA_d[cfreq] = num / den

        print(f"Elapsed: {time.time() - t_start:.4f}s")
        
        STL_ERA_d = 10 * np.log10(1.0 / (tau_ERA_d + 1e-20))

    ## Post-processing
    # Plot STLs
    plt.figure()
    for i in range(len(angles)):
        plt.plot(freq, STL_ERA[i, :])
    plt.ylim([0, 100])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Sound Transmission Loss [dB]')
    
    # Plot diffuse STL
    if STL_ERA_d is not None:
        plt.figure()
        plt.plot(freq, STL_ERA_d, 'k-', linewidth=1)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Diffuse Sound Transmission Loss [dB]')
        
        # Compare diffuse STL to analytical estimate
        # Pellecier and Trompette 2007
        STL_an_n = 10 * np.log10(1 + (omega * rho * h / (2 * rho0 * c0))**2)
        STL_an_d = STL_an_n - 5
        
        plt.figure()
        plt.plot(freq, STL_ERA_d, 'k-', linewidth=1, label='ERA')
        plt.plot(freq, STL_an_d, 'k--', label='Analytical est.')
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Diffuse Sound Transmission Loss [dB]')
    if fig_flag:
        plt.show()
    return freq, STL_ERA_d, tau_ERA_d

if __name__ == '__main__':
    print(main(fig_flag=False))