###################################################
# Import of all necessary packages
###################################################

# numpy: our main numerical package
import numpy as np
from tracking_library import *

###################################################
# Definition for thick quadrupoles
###################################################

# Modelling of a thick quadrupole
def Qthick(k1, l):
    '''Returns a thick quadrupole element (2x2 case)'''
    if k1>0:
        matrix = np.array([[np.cos(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sin(np.sqrt(k1)*l)],
                           [-np.sqrt(k1)*np.sin(np.sqrt(k1)*l), np.cos(np.sqrt(k1)*l)]])
    else:
        k1 = -k1
        matrix = np.array([[np.cosh(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sinh(np.sqrt(k1)*l)],
                           [np.sqrt(k1)*np.sinh(np.sqrt(k1)*l), np.cosh(np.sqrt(k1)*l)]])
    return  [{'matrix': matrix, 'length': l}]


# Modelling of a thick quadrupole with energy effects
def Qthick3(k1, l):
    '''Returns a thick quadrupole element (3x3 case)'''
    if k1>0:
        matrix = np.array([[np.cos(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sin(np.sqrt(k1)*l), 0],\
                           [-np.sqrt(k1)*np.sin(np.sqrt(k1)*l), np.cos(np.sqrt(k1)*l), 0],\
                           [0,0,1]])
    else:
        k1 = -k1
        matrix = np.array([[np.cosh(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sinh(np.sqrt(k1)*l), 0],\
                           [np.sqrt(k1)*np.sinh(np.sqrt(k1)*l), np.cosh(np.sqrt(k1)*l), 0],\
                           [0,0,1]])
    return [{'matrix': matrix, 'length': l}]


###################################################
# Definition of 4D elements, in cluding solenodis
###################################################

def Qthick4(k1, l):
    '''Returns a thick quadrupole element (4x4 case)'''
    if k1>0:
        matrix=np.array([[np.cos(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sin(np.sqrt(k1)*l), 0,0],
                         [-np.sqrt(k1)*np.sin(np.sqrt(k1)*l), np.cos(np.sqrt(k1)*l), 0, 0],
                         [0,0,np.cosh(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sinh(np.sqrt(k1)*l)],
                         [0,0,np.sqrt(k1)*np.sinh(np.sqrt(k1)*l), np.cosh(np.sqrt(k1)*l)],
                        ])
    else:
        k1=-k1
        matrix=np.array([[np.cosh(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sinh(np.sqrt(k1)*l), 0,0],
                         [np.sqrt(k1)*np.sinh(np.sqrt(k1)*l), np.cosh(np.sqrt(k1)*l), 0,0],
                         [0,0,np.cos(np.sqrt(k1)*l), 1/np.sqrt(k1)*np.sin(np.sqrt(k1)*l)],
                         [0,0,-np.sqrt(k1)*np.sin(np.sqrt(k1)*l), np.cos(np.sqrt(k1)*l)],
                         ])
    return [{'matrix': matrix, 'length': l}]


def D4(l):
    '''Returns a drift (4x4 case)'''
    matrix = np.array([[1, l, 0, 0],
                       [0, 1, 0, 0], 
                       [0, 0, 1, l], 
                       [0, 0, 0, 1]])    
    return [{'matrix': matrix, 'length': l}]

def Q4(f):
    '''Returns a quadrupole of focal length f (4x4 case)'''
    matrix = np.array([[1, 0, 0, 0],
                       [-1/f, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 1/f, 1]])
    
    return [{'matrix': matrix, 'length': 0}]

def B4(phi, l):
    '''Returns a sector bend of angle phi, and length l (4x4 case)'''
    matrix = np.array([[np.cos(phi),l/phi*np.sin(phi), 0, 0],
                        [-np.sin(phi)/l*phi, np.cos(phi), 0, 0],
                        [0, 0, 1, l],
                        [0, 0, 0, 1]])
    
    return [{'matrix': matrix, 'length': l}]




def solenoid(Bs, L, B_rho, q):
    '''Returns the L-long solenoid element with field Bs 
    normalized to B_rho and to the q polarity.'''
    K = np.sign(q)*Bs/B_rho/2
    C = np.cos(K*L)
    S = np.sin(K*L)
    matrix = np.array([[C**2, S*C/K, S*C, S**2/K],
                     [-K*S*C, C**2, -K*S**2, S*C],
                     [-S*C, -S**2/K, C**2, S*C/K],
                     [K*S**2, -S*C, -K*S*C, C**2]])
    return [{'matrix': matrix,'length': L}]


###################################################
# Transport 
###################################################

# we need a minor update to the transportParticles
def transportParticles4D(X_0, beamline, s_0=0):
    coords = [X_0]
    s = [s_0]
    for element in beamline:
        coords.append(element['matrix'] @ coords[-1])
        s.append(s[-1] + element['length']) 
    coords = np.array(coords)
    s = np.array(s)
    if np.shape(X_0)[0]<4:
        return {'x':  coords[:,0,:], # [s_idx, particle_idx]
                'xp': coords[:,1,:], # [s_idx, particle_idx]
                's':  s,             # [s_idx]
                'coords': coords,}   # [s_idx, coord_idx, particle_idx]
    elif np.shape(X_0)[0]==4:
        return {'x':  coords[:,0,:], # [s_idx, particle_idx]
            'xp': coords[:,1,:],     # [s_idx, particle_idx]
            'y':  coords[:,2,:],     # [s_idx, particle_idx]
            'py': coords[:,3,:],     # [s_idx, particle_idx]
            's':  s,                 # [s_idx]
            'coords': coords,}       # [s_idx, coord_idx, particle_idx]


# we need a minor update to the transportSigmas4D
def transportSigmas4D(sigma_0, beamline):
    sigmas = [sigma_0]
    s = [0]
    for element in beamline:
        sigmas.append(element['matrix'] @ sigmas[-1] @ element['matrix'].transpose())
        s.append(s[-1] + element['length']) 
    sigmas = np.array(sigmas)
    s = np.array(s)
    if len(sigma_0) < 4:
        return {'sigma11': sigmas[:, 0, 0],
                'sigma12': sigmas[:, 0, 1],
                'sigma21': sigmas[:, 1, 0], # equal to sigma12
                'sigma22': sigmas[:, 1, 1],
                's':  s,
                'sigmas': sigmas,}
    elif len(sigma_0)==4:
        return {'sigma11': sigmas[:, 0, 0],
                'sigma12': sigmas[:, 0, 1],
                'sigma21': sigmas[:, 1, 0], # equal to sigma12
                'sigma22': sigmas[:, 1, 1],
                'sigma33': sigmas[:, 2, 2],
                'sigma34': sigmas[:, 2, 3],
                'sigma43': sigmas[:, 3, 2], # equal to sigma34
                'sigma44': sigmas[:, 3, 3],
                's': s,
                'coords': sigmas,}
    

# Also updating the Twiss function
def twiss4D(beamline):
    '''Returns the Q, and the Twiss parameters beta, alpha, gamma of the beamline'''
    OTM = getEquivalentElement(beamline)
    R = OTM[0]['matrix']

    # 2x2 case
    if np.shape(R)[0]<4:
        # check that this matrix is stable:
        if np.abs(0.5*(R[0,0]+R[1,1])) > 1:
            raise ValueError('This beamline is not stable!')
        
        mu = np.arccos(0.5*(R[0,0]+R[1,1]))
        if (R[0,1]<0): 
            mu = 2*np.pi-mu
        tune = mu/(2*np.pi)
        beta = R[0,1]/np.sin(mu)
        alpha = (0.5*(R[0,0]-R[1,1]))/np.sin(mu)
        gamma = (1+alpha**2)/beta
        return tune, beta, alpha, gamma

    
    # 4x4 case, we assume uncoupled motion!!!
    if np.shape(R)[0]==4:
        Rx = R[:2,:2]
        # check that this matrix is stable:
        if np.abs(0.5*(Rx[0,0]+Rx[1,1])) > 1:
            raise ValueError('This beamline is not stable!')

        mux = np.arccos(0.5*(Rx[0,0]+Rx[1,1]))
        if (Rx[0,1]<0): 
            mux = 2*np.pi-mux
        Qx = mux/(2*np.pi)
        betax = Rx[0,1]/np.sin(mux)
        alphax = (0.5*(Rx[0,0]-Rx[1,1]))/np.sin(mux)
        gammax = (1+alphax**2)/betax
        
        Ry = R[2:,2:]
        # check that this matrix is stable:
        if np.abs(0.5*(Ry[0,0]+Ry[1,1])) > 1:
            raise ValueError('This beamline is not stable!')
        
        muy = np.arccos(0.5*(Ry[0,0]+Ry[1,1]))
        if (Ry[0,1]<0): 
            muy = 2*np.pi-muy
        Qy = muy/(2*np.pi)
        betay = Ry[0,1]/np.sin(muy)
        alphay = (0.5*(Ry[0,0]-Ry[1,1]))/np.sin(muy)
        gammay = (1+alphay**2)/betay
        
        return (Qx, betax, alphax, gammax, Qy, betay, alphay, gammay)
    
       