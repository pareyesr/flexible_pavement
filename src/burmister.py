import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d
from scipy.special import j0 as besselj0, j1 as besselj1
from Logica import Section

# Standard Axle Parameters
STANDARD_TIRE_RADIUS = 0.125  # m
STANDARD_TIRE_DISTANCE = 0.375  # m (center to center)
STANDARD_TIRE_PRESSURE = 0.662  # MPa
STANDARD_SUBGRADE_E = 50.0  # MPa (default, should be overridden)
STANDARD_SUBGRADE_V = 0.35  # Poisson ratio for subgrade

# Matrix functions - exactly matching MATLAB
def Matrix0(m, l, v):
    m_l = m * l
    exp_ml = np.exp(-m_l)
    return np.array([
        [exp_ml,  1, -(1 - 2*v) * exp_ml, 1 - 2*v],
        [exp_ml, -1,      2*v * exp_ml,   2*v]
    ])

def MatrixI(F, v, m, l, Est):
    m_l = m * l
    if Est == 0: # Bonded
        return np.array([
            [1,  F, -(1 - 2*v - m_l),  (1 - 2*v + m_l) * F],
            [1, -F,      2*v + m_l,     (2*v - m_l) * F],
            [1,  F,      1 + m_l,      -(1 - m_l) * F],
            [1, -F, -(2 - 4*v - m_l), -(2 - 4*v + m_l) * F]
        ])
    else: # Unbonded (Est == 1)
        return np.array([
            [1,  F, -(1 - 2*v - m_l),  (1 - 2*v + m_l) * F],
            [1, -F, -(2 - 4*v - m_l), -(2 - 4*v + m_l) * F],
            [1, -F,      2*v + m_l,     (2*v - m_l) * F],
            [0,  0,             0,               0]
        ])

def MatrixJ(R, F, v, m, l, Est):
    m_l = m * l
    if Est == 0: # Bonded
        return np.array([
            [  F,  1,   -(1 - 2*v - m_l) * F,      1 - 2*v + m_l],
            [  F, -1,       (2*v + m_l) * F,        2*v - m_l],
            [R*F,  R,       (1 + m_l) * R*F,     -(1 - m_l) * R],
            [R*F, -R, -(2 - 4*v - m_l) * R*F, -(2 - 4*v + m_l) * R]
        ])
    else: # Unbonded (Est == 1)
        return np.array([
            [  F,  1,   -(1 - 2*v - m_l) * F,      1 - 2*v + m_l],
            [R*F, -R, -(2 - 4*v - m_l) * R*F, -(2 - 4*v + m_l) * R],
            [  0,  0,                 0,               0],
            [  F, -1,       (2*v + m_l) * F,        2*v - m_l]
        ])

def MatrixS(v, m, l, R, Est):
    m_l = m * l
    if Est == 0: # Bonded
        return np.array([
            [ 1,        1 - 2*v + m_l],
            [-1,          2*v - m_l],
            [ R,       -(1 - m_l) * R],
            [-R, -(2 - 4*v + m_l) * R]
        ])
    else: # Unbonded (Est == 1)
        return np.array([
            [ 1,      1 - 2*v + m_l],
            [-R, -(2 - 4*v + m_l) * R],
            [ 0,               0],
            [-1,        2*v - m_l]
        ])

def EcTDef(m, p, v, E, A, B, C, D, J0, J1, L, Fa, Fb, Ht):
    m_L = m * L
    SZ  = -m*J0*((A-C*(1-2*v-m_L))*Fa + (B+D*(1-2*v+m_L))*Fb)
    term1 = (A+C*(1+m_L))*Fa + (B-D*(1-m_L))*Fb
    term2 = C*Fa - D*Fb
    SP  = (m*J0-J1/p)*term1 + 2*v*m*J0*term2
    ST  = J1/p*term1 + 2*v*m*J0*term2
    Tau = m*J1*((A+C*(2*v+m_L))*Fa-(B-D*(2*v-m_L))*Fb)
    W   = -((1+v)/E)*Ht*J0*((A-C*(2-4*v-m_L))*Fa-(B+D*(2-4*v+m_L))*Fb)
    U   = ((1+v)/E)*Ht*J1*term1
    return np.array([SZ, SP, ST, Tau, W, U])

def EcTDefS(m, p, v, E, B, D, J0, J1, L, Fb, Ht):
    m_L = m * L
    SZ  = -m*J0*((B+D*(1-2*v+m_L))*Fb)
    term1 = (B-D*(1-m_L))*Fb
    term2 = -D*Fb
    SP  = (m*J0-J1/p)*term1 + 2*v*m*J0*term2
    ST  = J1/p*term1 + 2*v*m*J0*term2
    Tau = m*J1*(-(B-D*(2*v-m_L))*Fb)
    W   = -((1+v)/E)*Ht*J0*(-(B+D*(2-4*v+m_L))*Fb)
    U   = ((1+v)/E)*Ht*J1*term1
    return np.array([SZ, SP, ST, Tau, W, U])

def Epsilon(E, Sz, Sr, St, v):
    ez = (1/E)*(Sz-v*(Sr+St))
    er = (1/E)*(Sr-v*(St+Sz))
    et = (1/E)*(St-v*(Sz+Sr))
    return np.array([ez, er, et])

def Ffactor(m, alfai, alfai_1):
    return np.exp(-m*(alfai-alfai_1))

def Rfactor(Ei, Ei1, vi1, vi):
    return (Ei/Ei1)*((1+vi1)/(1+vi))

def MatrixRot(S, gamma):
    c2 = np.cos(gamma)**2
    s2 = np.sin(gamma)**2
    sc = np.sin(gamma)
    cc = np.cos(gamma)
    s2g = np.sin(2*gamma)
    
    A = np.array([
        [0,  c2,    s2,   0,   0,   0,    0],
        [0,  s2,    c2,   0,   0,   0,    0],
        [1,   0,     0,   0,   0,   0,    0],
        [0,   0,     0,  sc,   0,   0,    0],
        [0,   0,     0,  cc,   0,   0,    0],
        [0, 0.5*s2g, -0.5*s2g, 0, 0,   0,    0],
        [0,   0,     0,   0,   0,  c2,   s2],
        [0,   0,     0,   0,   0,  s2,   c2],
        [0,   0,     0,   0,   1,   0,    0]
    ])
    M = A @ S
    return M.T

def compute_stresses_strains(nL, Thicknesslayers_orig, Interfaces_orig, Modulus_orig, ModulusP_orig, 
                             subgrade_E, subgrade_v, nT, Xc_tires, Yc_tires, Rc, QL, Xp, Yp, 
                             plot_geometry=False):
    """
    Compute stresses and strains for given parameters.
    Returns the results matrix Mxyz.
    """
    # --- Set up problem parameters ---
    H = np.sum(Thicknesslayers_orig)
    
    Modulus = np.append(Modulus_orig, subgrade_E)
    ModulusP = np.append(ModulusP_orig, subgrade_v)
    Interfaces = np.append(Interfaces_orig, 0) 

    # Add small offset to prevent numerical issues when analysis point is at tire center
    X_analysis = Xp + 1e-6
    Y_analysis = Yp + 1e-6

    # --- Plotting the geometry (optional) ---
    if plot_geometry:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(nT):
            p = Circle((Xc_tires[i], Yc_tires[i]), Rc[i], fill=False, color='blue', label=f'Tire {i+1}')
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        zlayer = 0
        ax.plot([X_analysis], [Y_analysis], [zlayer], 'r+', markersize=10, label="Analysis Points")
        for i in range(nL):
            zlayer -= Thicknesslayers_orig[i]
            ax.plot([X_analysis], [Y_analysis], [zlayer], 'r+', markersize=10)
        ax.set_xlabel('X Coordinate (m)'); ax.set_ylabel('Y Coordinate (m)'); ax.set_zlabel('Depth (m)')
        ax.set_title('Pavement Structure and Loading')
        if nT == 1: ax.legend()
        plt.show()

    # --- Pre-computation of dimensionless variables ---
    d = np.sqrt((X_analysis - Xc_tires)**2 + (Y_analysis - Yc_tires)**2)
    the = np.arctan2(Y_analysis - Yc_tires, X_analysis - Xc_tires)
    
    if H < 0.2:
        Nm = 30000
    else:
        Nm = 20000
        
    if H < 0.2:
        dm = 0.1
    elif H >= 1.1:
        dm = 1.0
    else:
        dm = H - 0.1
    
    Nv = 4 * (nL + 1) - 2
    NvR = 2 * nL + 1
    
    Mxyz = np.zeros((NvR, 9))
    Wy = np.zeros(NvR)

    Lambdas = np.cumsum(Thicknesslayers_orig) / H
    
    R = np.zeros(nL + 1)
    for t in range(nL + 1):
        E_i = Modulus[t]; nu_i = ModulusP[t]
        E_i1 = Modulus[t+1] if t < nL else Modulus[t]
        nu_i1 = ModulusP[t+1] if t < nL else ModulusP[t]
        R[t] = Rfactor(E_i, E_i1, nu_i1, nu_i)

    # --- Main Computational Loops ---
    for w in range(nT):
        b = np.zeros(Nv); b[0] = 1.0
        MatrixF = np.zeros((NvR, 6))

        q = QL[w]; a = Rc[w]
        p = d[w] if d[w] > 1e-9 else 1e-9
        ph = p / H; alfa = a / H
        
        # Integration loop - following MATLAB exactly
        for i in range(1, int(Nm) + 1):
            m = i * dm
            
            F = np.zeros(nL)
            F[0] = Ffactor(m, Lambdas[0], 0)
            for c in range(1, nL):
                F[c] = Ffactor(m, Lambdas[c], Lambdas[c-1])

            Mat = np.zeros((Nv, Nv))
            M1 = Matrix0(m, Lambdas[0], ModulusP[0])
            Mat[0:2, 0:4] = M1
            
            # Matrix assembly - following MATLAB indexing exactly
            for i_layer in range(nL):
                Mi = MatrixI(F[i_layer], ModulusP[i_layer], m, Lambdas[i_layer], Interfaces[i_layer])
                start_row = 4*i_layer + 2  # MATLAB: 4*i-1:4*i+2
                end_row = 4*i_layer + 6    # 4 rows
                start_col = 4*i_layer       # MATLAB: 4*i-3:4*i
                end_col = 4*i_layer + 4     # 4 columns
                Mat[start_row:end_row, start_col:end_col] = Mi

            for j_layer in range(nL-1):
                Mj = MatrixJ(R[j_layer], F[j_layer+1], ModulusP[j_layer+1], m, Lambdas[j_layer], Interfaces[j_layer])
                start_row = 4*j_layer + 2   # MATLAB: 4*j-1:4*j+2
                end_row = 4*j_layer + 6    # 4 rows
                start_col = 4*j_layer + 4  # MATLAB: 4*j+1:4*j+4
                end_col = 4*j_layer + 8    # 4 columns
                Mat[start_row:end_row, start_col:end_col] = -Mj

            MS = MatrixS(ModulusP[nL], m, Lambdas[nL-1], R[nL-1], Interfaces[nL-1])
            Mat[Nv-4:Nv, Nv-2:Nv] = -MS
            
            try:
                AA = np.linalg.solve(Mat, b)
            except np.linalg.LinAlgError:
                continue

            MatrixR = np.zeros((NvR, 6))
            J0 = besselj0(m * ph)
            J1 = besselj1(m * ph)
            
            # First Layer - Surface (z=0)
            MatrixR[0, :] = EcTDef(m, ph, ModulusP[0], Modulus[0], AA[0], AA[1], AA[2], AA[3], J0, J1, 0, F[0], 1, H)
            
            # Intermediate layers - Upper interfaces
            for i in range(1, nL):
                idx = 4 * i
                L_val = Lambdas[i-1]; Fa_val = F[i]; Fb_val = 1.0
                MatrixR[2*i, :] = EcTDef(m, ph, ModulusP[i], Modulus[i], AA[idx], AA[idx+1], AA[idx+2], AA[idx+3], J0, J1, L_val, Fa_val, Fb_val, H)
            
            # Lower interfaces
            for j in range(nL):
                idx = 4 * j
                L_val = Lambdas[j]; Fa_val = 1.0; Fb_val = F[j]
                MatrixR[2*j + 1, :] = EcTDef(m, ph, ModulusP[j], Modulus[j], AA[idx], AA[idx+1], AA[idx+2], AA[idx+3], J0, J1, L_val, Fa_val, Fb_val, H)
            
            # Subgrade
            MatrixR[NvR-1, :] = EcTDefS(m, ph, ModulusP[nL], Modulus[nL], AA[Nv-2], AA[Nv-1], J0, J1, Lambdas[nL-1], 1, H)

            J1a = besselj1(m * alfa) * dm / m
            if not np.isnan(m) and m > 0:
                 MatrixF += (MatrixR * J1a)

        MatrixF *= (q * alfa)
        
        FMatrix_with_strains = np.zeros((NvR, 9))
        FMatrix_with_strains[:, :6] = MatrixF

        # Computing strains - corrected indexing
        for i in range(nL):
            mod = Modulus[i]; poi = ModulusP[i]
            # Bottom of layer i
            FMatrix_with_strains[2*i+1, 6:9] = Epsilon(mod, MatrixF[2*i+1,0], MatrixF[2*i+1,1], MatrixF[2*i+1,2], poi) * 1e6
        
        for j in range(nL + 1):
            mod = Modulus[j]; poi = ModulusP[j]
            # Top of layer j
            FMatrix_with_strains[2*j, 6:9] = Epsilon(mod, MatrixF[2*j,0], MatrixF[2*j,1], MatrixF[2*j,2], poi) * 1e6

        Wy += MatrixF[:, 4]
        FMatrix = np.delete(FMatrix_with_strains, [4, 5], axis=1)
        
        RMatrix = np.zeros((NvR, 9))
        for e in range(NvR):
            S = FMatrix[e, :]
            RMatrix[e, :] = MatrixRot(S, -the[w])
        
        if X_analysis < Xc_tires[w]:
             RMatrix[:, [3, 4]] *= -1
        
        Mxyz += RMatrix

    # --- Finalize Results ---
    Mxyz = np.insert(Mxyz, 9, Wy * 1e5, axis=1)
    Mxyz = np.round(Mxyz, 3)
    
    # Replace any remaining NaN/inf with 0 for cleaner output
    Mxyz = np.where(np.isfinite(Mxyz), Mxyz, 0)
    
    return Mxyz

def analyze_section(section: Section, subgrade_E: float = STANDARD_SUBGRADE_E, 
                     subgrade_v: float = STANDARD_SUBGRADE_V, plot_geometry: bool = False,
                     bonded=True):
    """
    Analyze a pavement section using standard axle parameters.
    
    Parameters:
    -----------
    section : Section
        A Section object containing Layer objects with material properties
    subgrade_E : float
        Subgrade Young's modulus in MPa (default: 50.0)
    subgrade_v : float
        Subgrade Poisson ratio (default: 0.35)
    plot_geometry : bool
        Whether to plot the geometry (default: False)
    bonded : bool | list[bool] | dict[int, bool]
        Interface bonding condition. True means bonded (default), False sets all interfaces
        to unbonded. Passing a list/tuple/array assigns bonding per layer (length >= nL),
        and a dict maps layer index to bonding flag.
    
    Returns:
    --------
    dict : Dictionary with maximum values and detailed results for all analysis points
    """
    # Convert thickness from inches to meters (1 inch = 0.0254 m)
    INCH_TO_METER = 0.0254
    
    # Extract layer data from Section
    nL = len(section)
    Thicknesslayers_orig = np.array([layer.thickness * INCH_TO_METER for layer in section])
    Modulus_orig = np.array([layer.E for layer in section])
    ModulusP_orig = np.array([layer.Poisson_ratio for layer in section])
    
    # Validate that all moduli are valid (not NaN)
    if np.any(np.isnan(Modulus_orig)):
        missing_moduli = [i for i, layer in enumerate(section) if np.isnan(layer.E)]
        layer_names = [section[i].name for i in missing_moduli]
        raise ValueError(f"Missing Young's Modulus (E) values for layers: {layer_names}. "
                        f"Please ensure all layers have valid E values in the material database.")
    
    # Validate that all Poisson ratios are valid
    if np.any(np.isnan(ModulusP_orig)):
        missing_poisson = [i for i, layer in enumerate(section) if np.isnan(layer.Poisson_ratio)]
        layer_names = [section[i].name for i in missing_poisson]
        raise ValueError(f"Missing Poisson ratio values for layers: {layer_names}. "
                        f"This should not happen as Poisson ratios are set automatically based on material type.")
    
    # Set interfaces (0 bonded, 1 unbonded)
    Interfaces_orig = np.zeros(nL, dtype=int)

    if isinstance(bonded, bool):
        if not bonded:
            Interfaces_orig[:] = 1
    elif isinstance(bonded, (list, tuple, np.ndarray)):
        for idx, flag in enumerate(bonded):
            if idx >= nL:
                break
            Interfaces_orig[idx] = 0 if flag else 1
    elif isinstance(bonded, dict):
        for idx, flag in bonded.items():
            if 0 <= idx < nL:
                Interfaces_orig[idx] = 0 if flag else 1

    # Ensure subgrade-marked layers remain unbonded
    for i, layer in enumerate(section):
        if layer.subgrade_code == 1:
            Interfaces_orig[i] = 1
    
    # Standard axle configuration: 2 tires
    nT = 2
    tire_radius = STANDARD_TIRE_RADIUS
    tire_distance = STANDARD_TIRE_DISTANCE
    tire_pressure = STANDARD_TIRE_PRESSURE
    
    # Position tires symmetrically about origin
    # Tire 1: left side, Tire 2: right side
    Xc_tires = np.array([-tire_distance / 2, tire_distance / 2])
    Yc_tires = np.array([0.0, 0.0])
    Rc = np.array([tire_radius, tire_radius])
    QL = np.array([tire_pressure, tire_pressure])
    
    # Define three analysis points
    # 1. Center of left tire
    Xp1 = Xc_tires[0]
    Yp1 = Yc_tires[0]
    
    # 2. Center between both tires (middle)
    Xp2 = 0.0
    Yp2 = 0.0
    
    # 3. Internal edge of left tire (closest to the other tire)
    Xp3 = Xc_tires[0] + tire_radius  # Edge at the right side of left tire
    Yp3 = 0.0
    
    analysis_points = [
        ("Center of Tire", Xp1, Yp1),
        ("Center Between Tires", Xp2, Yp2),
        ("Internal Edge of Tire", Xp3, Yp3)
    ]
    
    # Compute results for each analysis point
    all_results = []
    for point_name, Xp, Yp in analysis_points:
        Mxyz = compute_stresses_strains(
            nL, Thicknesslayers_orig, Interfaces_orig, Modulus_orig, ModulusP_orig,
            subgrade_E, subgrade_v, nT, Xc_tires, Yc_tires, Rc, QL, Xp, Yp, 
            plot_geometry=(plot_geometry and point_name == "Center of Tire")  # Only plot once
        )
        all_results.append((point_name, Xp, Yp, Mxyz))
    
    # Find maximum values across all analysis points
    # Stack all results: shape will be (3 analysis points, NvR, 10)
    all_Mxyz = np.stack([Mxyz for _, _, _, Mxyz in all_results], axis=0)
    
    # For maximum, find the point with maximum absolute value for each component
    # This gives us the worst-case scenario (largest magnitude)
    abs_Mxyz = np.abs(all_Mxyz)
    max_indices = np.argmax(abs_Mxyz, axis=0)
    
    # Extract the actual signed values at the maximum magnitude locations
    n_rows, n_cols = all_Mxyz.shape[1], all_Mxyz.shape[2]
    max_Mxyz = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            max_Mxyz[i, j] = all_Mxyz[max_indices[i, j], i, j]
    
    # Create row and column names
    row_names = []
    for i in range(nL):
        row_names.append(f"Layer {i+1} Top")
        row_names.append(f"Layer {i+1} Bottom")
    row_names.append(f"Subgrade Top")
    
    col_names = ['SigmaX', 'SigmaY', 'SigmaZ', 'TauXY', 'TauZY', 'TauZX', 'EpsX', 'EpsY', 'EpsZ', 'W (10^-5 m)']
    
    # Create DataFrame for maximum values
    max_df = pd.DataFrame(max_Mxyz, index=row_names, columns=col_names)
    
    # Create detailed results for each point
    detailed_results = {}
    for point_name, Xp, Yp, Mxyz in all_results:
        df = pd.DataFrame(Mxyz, index=row_names, columns=col_names)
        detailed_results[point_name] = {
            'x': Xp,
            'y': Yp,
            'results': df
        }
    
    return {
        'max_values': max_df,
        'detailed_results': detailed_results,
        'section_info': section.info()
    }
