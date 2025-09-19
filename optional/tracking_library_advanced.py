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
    print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
    matrix = []
    # YOUR IMPLEMENTATION HERE

    return  [{'matrix': matrix, 'length': l}]


# Modelling of a thick quadrupole with energy effects
def Qthick3(k1, l):
    '''Returns a thick quadrupole element (3x3 case)'''
    print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
    matrix = []
    # YOUR IMPLEMENTATION HERE

    return [{'matrix': matrix, 'length': l}]


###################################################
# Definition of 4D elements, in cluding solenodis
###################################################

def Qthick4(k1, l):
    '''Returns a thick quadrupole element (4x4 case)'''
    print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
    matrix = []
    # YOUR IMPLEMENTATION HERE

    return [{'matrix': matrix, 'length': l}]

def D4(l):
    '''Returns a drift (4x4 case)'''
    print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
    matrix = []
    # YOUR IMPLEMENTATION HERE
    
    return [{'matrix': matrix, 'length': l}]

def Q4(f):
    '''Returns a quadrupole of focal length f (4x4 case)'''
    print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
    matrix = []
    # YOUR IMPLEMENTATION HERE
    
    return [{'matrix': matrix, 'length': 0}]

def B4(phi, l):
    '''Returns a sector bend of angle phi, and length l (4x4 case)'''
    print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
    matrix = []
    # YOUR IMPLEMENTATION HERE
    
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
        print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
        return {'x':  [],# YOUR IMPLEMENTATION HERE
            'xp': [],# YOUR IMPLEMENTATION HERE
            'y':  [],# YOUR IMPLEMENTATION HERE
            'py': [],# YOUR IMPLEMENTATION HERE
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
        print('NOT YET IMPLEMENTED! Please, edit me in `tracking_library_advanced.py` ... ')
        return {'sigma11': sigmas[:, 0, 0],
                'sigma12': sigmas[:, 0, 1],
                'sigma21': sigmas[:, 1, 0], # equal to sigma12
                'sigma22': sigmas[:, 1, 1],
                'sigma33': [],# YOUR IMPLEMENTATION HERE
                'sigma34': [],# YOUR IMPLEMENTATION HERE
                'sigma43': [],# YOUR IMPLEMENTATION HERE
                'sigma44': [],# YOUR IMPLEMENTATION HERE
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
        Q = mu/(2*np.pi)
        beta = R[0,1]/np.sin(mu)
        alpha = (0.5*(R[0,0]-R[1,1]))/np.sin(mu)
        gamma = (1+alpha**2)/beta
        return Q, beta, alpha, gamma

    
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
        
        
        # YOUR IMPLEMENTATION HERE
        Qy = ''
        betay = ''
        alphay = ''
        gammay = ''
        
                
        return (Qx, betax, alphax, gammax, Qy, betay, alphay, gammay)
    
###################################################
# Other useful systems
###################################################

def particle_emittance(x, xp, beta, alpha):
    '''Returns the single particle emittance for a given 
       x, xp particle coordinates and assumed beta and alpha parameters
    '''
    # compute gamma
    gamma = (alpha**2 + 1)/beta
    # compute and return the associated single particle emittance
    epsilon = gamma*x**2 + 2*alpha*x*xp + beta*xp**2
    return epsilon

def ellipse_points(emittance, beta, alpha, n_points = 100):
    ''' Returns the x,x' coordinates of an ellipse in phase space for 
        the given Twiss parameters (beta, gamma, emittance)
    '''
    # generate a uniform sampling of phases:
    thetas = np.linspace(0, 2*np.pi, n_points)
    # generate coordinates
    x  = np.sqrt(emittance*beta)*np.cos(thetas)
    xp = -np.sqrt(emittance/beta)*(alpha*np.cos(thetas)-np.sin(thetas))
    # return them in our usual form
    return np.array([x, xp])

def beam(emittance, beta, alpha, n_particles = 100):
    ''' Returns the x,x' coordinates of Gaussian beam matched to 
        the given Twiss parameters (beta, gamma, emittance)
    '''
    gamma = (1+alpha**2)/beta
    sigma_matrix = np.array(
        [[ beta*emittance, -alpha*emittance],
         [-alpha*emittance, gamma*emittance]])

    return np.random.multivariate_normal((0, 0), sigma_matrix, n_particles).transpose()