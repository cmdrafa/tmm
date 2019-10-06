import numpy as np


class dat:
    pass


def tmm1d(DEV, SRC):
    """ 
    This module uses the transfer matrix method for a multilayer structure to calculate the S-Matrix,
    the reflection and transmission coefficients. 

    Parameters
    ----------
    DEV -> must be a object, similar to a Matlab struct with the fields being:
    .er1 -> permitivity reflection interface
    .er2 -> permitivity transmission interface
    .ur1 -> permeability reflection interface
    .ur2 -> permeability transmission interface
    .ER -> Numpy array with the permitivity of the device layers
    .UR -> Numpy array with the permeability of device layers
    .L -> Numpy array with the length of the device layers

    SRC -> must be a object, similar to a Matlab struct with the fields being:
    .ptm -> Amplitude of the TM polarization
    .pte -> Amplitude of the TE polarization
    .phi -> Azimuthal angle of incidence (Degrees)
    .theta -> Elevation angle of incidence (Degrees) 
    .lam0 -> Free space wavelength
    ----------

    Return value
    ------------
    DAT -> object, similar to a Matlab struct with the fields being:
    .s11 -> s11 parameter
    .s12 -> s12 parameter
    .s21 -> s21 parameter
    .s22 -> s22 paramete
    .REF -> Refelection coefficient
    .TRN -> transmission coefficient
    
    """
    # Calculate transverse wave parameters
    SRC.theta = np.deg2rad(SRC.theta)
    SRC.phi = np.deg2rad(SRC.phi)
    DEV.L = DEV.L
    ninc = np.sqrt(DEV.ur1*DEV.er1)
    k0 = (2*np.pi) / SRC.lam0
    kx = ninc*np.sin(SRC.theta)*np.cos(SRC.phi)
    ky = ninc*np.sin(SRC.theta)*np.sin(SRC.phi)
    N_layers = len(DEV.UR)

    # Gap medium parameters
    Qg = np.array([[kx*ky, 1+pow(ky, 2)], [-(1+pow(kx, 2)), -(kx*ky)]])
    Vg = -1j * Qg

    # Global S-Matrix
    s11 = np.zeros((2, 2), dtype=complex)
    s12 = np.eye(2, 2, dtype=complex)
    s21 = np.eye(2, 2, dtype=complex)
    s22 = np.zeros((2, 2), dtype=complex)

    Q = np.zeros((2, 2, N_layers), dtype=complex, order='C')
    Omega = np.zeros((2, 2, N_layers), dtype=complex, order='C')
    V = np.zeros((2, 2, N_layers), dtype=complex, order='C')
    kz = np.zeros(2, dtype=complex, order='F')

    # Parameters for layer i
    for i in range(N_layers):
        kz[i] = np.sqrt(DEV.UR[i]*DEV.ER[i] - pow(kx, 2) - pow(ky, 2))
        Q[i, :, :] = [[kx*ky, DEV.UR[i]*DEV.ER[i] - pow(kx, 2)],
                      [pow(ky, 2) - DEV.UR[i]*DEV.ER[i], -(kx*ky)]]
        Q[i, :, :] = (1/DEV.UR[i]) * Q[i, :, :]
        Omega[i, :, :] = np.eye(2, 2)
        Omega[i, :, :] = Omega[i, :, :] * (1j*kz[i])
        V[i, :, :] = (Q[i, :, :] @ np.linalg.inv(Omega[i, :, :]))

    X = np.zeros((2, 2, N_layers), dtype=complex)
    A = np.zeros((2, 2, N_layers), dtype=complex)
    B = np.zeros((2, 2, N_layers), dtype=complex)
    D = np.zeros((2, 2, N_layers), dtype=complex)
    s11_layer = np.zeros((2, 2, N_layers), dtype=complex)
    s12_layer = np.zeros((2, 2, N_layers), dtype=complex)
    s21_layer = np.zeros((2, 2, N_layers), dtype=complex)
    s22_layer = np.zeros((2, 2, N_layers), dtype=complex)

    # Scattering matrix por layer i
    for i in range(N_layers):
        X[i, :, :] = np.diag(np.exp(np.diag(Omega[i, :, :]*k0*DEV.L[i])))
        A[i, :, :] = np.eye(2, 2) + (np.linalg.inv(V[i, :, :]) @ Vg)
        B[i, :, :] = np.eye(2, 2) - (np.linalg.inv(V[i, :, :]) @ Vg)
        D[i, :, :] = A[i, :, :] - X[i, :, :] @ B[i, :,
                                                 :] @ np.linalg.inv(A[i, :, :]) @ X[i, :, :] @ B[i, :, :]
        s11_layer[i, :, :] = np.linalg.inv(D[i, :, :])  @ (X[i, :, :] @ B[i, :, :] @ np.linalg.inv(A[i, :, :]) @
                                                           X[i, :, :] @ A[i, :, :] - B[i, :, :])
        s22_layer[i, :, :] = s11_layer[i, :, :]
        s12_layer[i, :, :] = np.linalg.inv(D[i, :, :]) @ X[i, :, :] @ (
            A[i, :, :] - B[i, :, :] @ np.linalg.inv(A[i, :, :]) @ B[i, :, :])
        s21_layer[i, :, :] = s12_layer[i, :, :]

    # Update Global matrix using Redheffer product
    D_2 = np.zeros((2, 2, N_layers), dtype=complex)
    F = np.zeros((2, 2, N_layers), dtype=complex)
    for i in range(N_layers):
        D_2[i, :, :] = s12 @ np.linalg.inv(
            np.eye(2, 2, dtype=complex) - (s11_layer[i, :, :] @ s22))
        F[i, :, :] = s21_layer[i, :, :] @ np.linalg.inv(
            np.eye(2, 2) - s22 @ s11_layer[i, :, :])
        s11 = s11 + D_2[i, :, :] @ s11_layer[i, :, :] @ s21
        s12 = D_2[i, :, :] @ s12_layer[i, :, :]
        s21 = F[i, :, :] @ s21
        s22 = s22_layer[i, :, :] + F[i, :, :] @ s22 @ s12_layer[i, :, :]

    # Connect to external regions
    # S-parameters reflected
    kz_ref = np.sqrt(DEV.ur1*DEV.er1 - pow(kx, 2) - pow(ky, 2))
    Q_ref = np.array([[kx*ky, DEV.ur1*DEV.er1 - pow(kx, 2)],
                      [pow(ky, 2) - DEV.ur1*DEV.er1, -(kx*ky)]])
    Q_ref = (1/DEV.ur1) * Q_ref
    Omega_ref = np.eye(2, 2)
    Omega_ref = (1j*kz_ref) * Omega_ref
    V_ref = Q_ref @ np.linalg.inv(Omega_ref)
    A_ref = np.eye(2, 2) + (np.linalg.inv(Vg) @ V_ref)
    B_ref = np.eye(2, 2) - (np.linalg.inv(Vg) @ V_ref)
    s11_ref = -1 * (np.linalg.inv(A_ref) @ B_ref)
    s12_ref = 2 * np.linalg.inv(A_ref)
    s21_ref = 0.5 * (A_ref - (B_ref @ np.linalg.inv(A_ref) @ B_ref))
    s22_ref = B_ref @ np.linalg.inv(A_ref)

    # Update global matrix with reflected parameters
    # Redheffer product
    D_ref = s12_ref @ np.linalg.inv(np.eye(2, 2) - s11 @ s22_ref)
    F_ref = s21 @ np.linalg.inv(np.eye(2, 2) - s22_ref @ s11)
    s22 = s22 + F_ref @ s22_ref @ s12
    s21 = (F_ref @ s21_ref)
    s12 = D_ref @ s12
    s11 = s11_ref + D_ref @ s11 @ s21_ref

    # S-parameters transmitted
    kz_trn = np.sqrt(DEV.ur2*DEV.er2 - pow(kx, 2) - pow(ky, 2))
    Q_trn = np.array([[kx*ky, DEV.ur2*DEV.er2 - pow(kx, 2)],
                      [pow(ky, 2) - DEV.ur2*DEV.er2, -(kx*ky)]])
    Q_trn = Q_trn * (1/DEV.ur2)
    Omega_trn = np.eye(2, 2)
    Omega_trn = (1j*kz_trn) * Omega_trn
    V_trn = Q_trn @ np.linalg.inv(Omega_trn)
    A_trn = np.eye(2, 2) + (np.linalg.inv(Vg) @ V_trn)
    B_trn = np.eye(2, 2) - (np.linalg.inv(Vg) @ V_trn)
    s11_trn = B_trn @ np.linalg.inv(A_trn)
    s12_trn = 0.5 * (A_trn - B_trn @ np.linalg.inv(A_trn) @ B_trn)
    s21_trn = 2 * np.linalg.inv(A_trn)
    s22_trn = -1 * (np.linalg.inv(A_trn) @ B_trn)

    # Update global matrix with transmitted s-parameters
    # Redheffer product
    D_trn = s12 @ np.linalg.inv(np.eye(2, 2) - s11_trn @ s22)
    F_trn = s21_trn @ np.linalg.inv(np.eye(2, 2) - s22@s11_trn)
    s11 = s11 + D_trn @ s11_trn @ s21
    s12 = D_trn @ s12_trn
    s21 = F_trn @ s21
    s22 = s22_trn + F_trn @ s22 @ s12_trn

    # Calculate Source Paramters
    kinc = np.array([
        np.sin(SRC.theta)*np.cos(SRC.phi),
        np.sin(SRC.theta)*np.sin(SRC.phi),
        np.cos(SRC.theta)])

    kinc = k0 * ninc * kinc
    n_hat = np.array([0, 0, 1])

    if SRC.theta == 0:
        a_te = np.array([0, n_hat[2], 0])
    else:
        a_te = np.cross(n_hat, kinc) / np.linalg.norm(np.cross(n_hat, kinc), 2)

    # Check Operator for this later
    a_tm = np.cross(a_te, kinc) / np.linalg.norm(np.cross(a_te, kinc), 2)

    P = SRC.pte * a_te + SRC.ptm*a_tm
    e_src = np.array([[P[0], P[1]]])

    ## Calculate Reflected and transmitted fields
    e_ref = s11 @ e_src.T
    e_trn = s21 @ e_src.T

    ## Calculate longitudinal field Components
    Ez_ref = - ( kx * e_ref[0] + ky * e_ref[1] ) / kz_ref
    Ez_trn = - ( kx * e_trn[0] + ky * e_trn[1] ) / kz_trn

    ## Calculate transmittance and Reflectance
    REF = pow(np.abs(e_ref[0]),2) + pow(np.abs(e_ref[1]),2) + pow(np.abs(Ez_ref),2)
    E_trn = pow(np.abs(e_trn[0]),2) + pow(np.abs(e_trn[1]),2) + pow(np.abs(Ez_trn),2)
    TRN = E_trn * np.real(kz_trn/DEV.ur2) / np.real(kz_ref / DEV.ur1)

    DAT = dat()

    DAT.s11 = s11
    DAT.s12 = s12
    DAT.s21 = s21
    DAT.s22 = s22
    DAT.REF = REF
    DAT.TRN = TRN

    return DAT
