import numpy as np
from tmm1d import tmm1d
import time

class Parameters:
    pass

def main():
    DEV = Parameters()

    ## Device Parameters
    DEV.er1 = 1.4 # permitivity reflection interface
    DEV.er2 = 1.8 # permitivity transmission interface
    DEV.ur1 = 1.2 # permeability reflection interface
    DEV.ur2 = 1.6 # permeability transmission interface
    DEV.ER = np.array([2,1]) # permitivity of the device layers (you can add any amout of layers you want)
    DEV.UR = np.array([1,3]) # permeability of device layers (you can add any amount of layers you want)
    DEV.L = np.array([0.25,0.5]) * 2.7 # Length of the layers
    

    ## Source Parameters
    SRC = Parameters()
    SRC.ptm = (1j)/np.sqrt(2) # Amplitude of the TM polarization
    SRC.pte = 1 / np.sqrt(2) # Amplitude of the TE polarization
    SRC.phi = 23 # Azimuthal angle of incidence (Degrees)
    SRC.theta = 57 # Elevation angle of incidence (Degrees) 
    SRC.lam0 = 2.7 # Free space wavelength 
    print(SRC.lam0)

    start_time = time.time()
    DAT = tmm1d(DEV, SRC)
    print('--- %s seconds ---' %(time.time() - start_time))

    print('REF: ', DAT.REF)
    print('TRN: ', DAT.TRN)

    print('S11:\n', DAT.s11)
    print('S12:\n', DAT.s12)
    print('S21:\n', DAT.s21)
    print('S22:\n', DAT.s22)



if __name__ == "__main__":
    main()