###################################################
# Import of all necessary packages
###################################################

# numpy: our main numerical package
import numpy as np
# matplotlib and seaborn: our plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# linear algebra and optimisation algorithms
from numpy.linalg import norm
from scipy.optimize import minimize
# some useful package
from copy import deepcopy

# **Optional:** for animations you might need to configure your jupyter lab properly:
# > pip install ipywidgets
# > jupyter nbextension enable --py widgetsnbextension
# > jupyter labextension install @jupyter-widgets/jupyterlab-manager
from ipywidgets import interactive

# some setup for the plotting
from matplotlib import pyplot as plt



###################################################
# Beamline definition and tracking functions
###################################################

# Modelling of a drift
def D(L):
    '''Returns a list containing a single "drift" of length L'''
    # NB: we return a list with a dict
    # the dict contains the matrix (the transformation)
    # and the element length 
    return [{'matrix':np.array([[1, L],[0, 1]]), 'length':L}] 


# Modelling of a quadrupole
def Q(f):
    '''Returns a list containing a quadrupole with focal length f'''
    # NB: we return a list with a dict
    # the dict contains the matrix (the transformation)
    # and the element length 
    return [{'matrix':np.array([[1, 0],[-1/f, 1]]), 'length':0}]


# From a list of elements - or beamline - to an equivalent single element
def getEquivalentElement(beamline):
    '''Returns the single element which is equivalent of the given beamline'''
    # we start from an identity matrix (np.eye)
    # with the same dimension of the matrix of the
    # first element of the beamline
    equivalentElement = np.eye(beamline[0]['matrix'].shape[0])
    length = 0
    # NB: we reverse the order of the beamline ([-1::-1])
    for elements in beamline[-1::-1]:
        # we multiply the matrices 
        equivalentElement = equivalentElement @ elements['matrix']
        # and we add the lengths
        length = length + elements['length']
    # we return the dict with the "usual" keys (matrix and length) embedded in a
    #  list (with a single element), as for the definition of the D and Q functions
    return [{'matrix':equivalentElement, 'length':length}]


# Tracking particles along a beamline
def transportParticles(X_0, beamline, s_0=0):
    '''Track the particle(s) `X_0` along the given `beamline`. 
    If needed, one can specify an initial longitudinal position `s_0`, otherwise set to 0.

    It will return a dictionary containing the following key:values
       'x': a NxM numpy array with the M-particles x position for all N-elements of the beamline
       'xp': a NxM numpy array with the M-particles x' angles for all N-elements of the beamline
       's': a N-long numpy array with the longitudinal position of the N-elements of the beamline
       'coords': a Nx2xM numpy array with all M-particles coordinates (both x and x') at all N-elements of the beamline
    
    Disclaimer: if beamline is made of 5 elements, the output will have 5+1 "elements" as it will also 
                return include the initial particle coordinates.
    '''
    coords = [X_0]
    s = [s_0]
    for element in beamline:
        coords.append(element['matrix'] @ coords[-1])
        s.append(s[-1] + element['length']) 
    coords = np.array(coords)
    s = np.array(s)
    return {'x':  coords[:,0,:], # [s_idx, particle_idx]
            'xp': coords[:,1,:], # [s_idx, particle_idx]
            's':  s,   # [s_idx]
            'coords': coords,}   # [s_idx, coord_idx, particle_idx]



###################################################
# Multi-particle tracking functions
###################################################

def transportSigmas(sigma_0, beamline):
    '''Transport the input sigma matrix (sigma_0) along the given beamline
    
    It will return a dictionary containing the following key:values
        'sigma11': a N-long numpy array with the \sigma_{11} value for all N-elements of the beamline
        'sigma12': a N-long numpy array with the \sigma_{12} value for all N-elements of the beamline
        'sigma21': a N-long numpy array with the \sigma_{21} value for all N-elements of the beamline
        'sigma22': a N-long numpy array with the \sigma_{22} value for all N-elements of the beamline
        's': a N-long numpy array with the longitudinal position of the N-elements of the beamline
        'sigmas': a Nx2x2 numpy array with all sigma matrices at all N-elements of the beamline
    
    Disclaimer: if beamline is made of 5 elements, the output will have 5+1 "elements" as it will also 
                return include the initial sigma matrix.
    '''

    sigmas = [sigma_0]
    s = [0]
    for element in beamline:
        sigmas.append(element['matrix'] @ sigmas[-1] @ element['matrix'].transpose())
        s.append(s[-1] + element['length']) 
    sigmas = np.array(sigmas)
    s = np.array(s)
    return {'sigma11': sigmas[:, 0, 0],
            'sigma12': sigmas[:, 0, 1],
            'sigma21': sigmas[:, 1, 0], # equal to sigma12
            'sigma22': sigmas[:, 1, 1],
            's':  s,
            'sigmas': sigmas,}


###################################################
# Periodic systems
###################################################

def twiss(beamline):
    '''
    Computes and returns the closed solution (if it exist!) Tune (Q) and Twiss parameters 
    beta, alpha, gamma of the given beamline.
    '''

    # first, compute the equivalent "One-Turn-Map", and extract its matrix:
    OTM = getEquivalentElement(beamline)
    R = OTM[0]['matrix']
    
    # check that this matrix is stable:
    if np.abs(0.5*(R[0,0]+R[1,1])) > 1:
        raise ValueError('This beamline is not stable!')
    
    # all relevant Twiss parameters can be extrcted from the matrix:
    mu = np.arccos(0.5*(R[0,0]+R[1,1]))
    if (R[0,1]<0): 
        mu = 2*np.pi-mu
    Q = mu/(2*np.pi)
    beta = R[0,1]/np.sin(mu)
    alpha = (0.5*(R[0,0]-R[1,1]))/np.sin(mu)
    gamma = (1+alpha**2)/beta
    
    return Q, beta, alpha, gamma



###################################################
# Function from 05_Advanced_Exercises.ipynb
###################################################

def Qthick(k1, l):
    '''Returns a thick quadrupole element'''
    if k1>0:
        matrix = np.array([[np.cos(np.sqrt(k1)*l), 
                          1/np.sqrt(k1)*np.sin(np.sqrt(k1)*l)],
                          [-np.sqrt(k1)*np.sin(np.sqrt(k1)*l), 
                          np.cos(np.sqrt(k1)*l)]])
    else:
        k1 = -k1
        matrix = np.array([[np.cosh(np.sqrt(k1)*l),
                          1/np.sqrt(k1)*np.sinh(np.sqrt(k1)*l)],
                          [np.sqrt(k1)*np.sinh(np.sqrt(k1)*l),
                          np.cosh(np.sqrt(k1)*l)]])
    return  [{'matrix': matrix, 'length': l}]


def B(phi, L):
    '''Returns a list containing a thick bend with and length L'''
    # NB: we return a list with a dict
    # the dict contains the matrix (the transformation)
    # and the element length 

    # compute the 2D bend matrix:
    bend_matrix = np.array(
        [[np.cos(phi),          L/phi*np.sin(phi)],
        [-np.sin(phi)/L*phi,    np.cos(phi)]]
        )

    return [{'matrix':bend_matrix, 'length':L}]


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


def R2beta(R):
    # 2x2 case
    if np.shape(R)[0]==2:
        mu = np.arccos(0.5*(R[0,0]+R[1,1]))
        if (R[0,1]<0): 
            mu = 2*np.pi-mu
        Q = mu/(2*np.pi)
        beta = R[0,1]/np.sin(mu)
        alpha = (0.5*(R[0,0]-R[1,1]))/np.sin(mu)
        gamma = (1+alpha**2)/beta
        return (Q, beta, alpha, gamma)
    
    # 3x3 case
    if np.shape(R)[0]==3:
        R = R[:3,:3]
        mu = np.arccos(0.5*(R[0,0]+R[1,1]))
        if (R[0,1]<0): 
            mu = 2*np.pi-mu
        Q = mu/(2*np.pi)
        beta = R[0,1]/np.sin(mu)
        alpha = (0.5*(R[0,0]-R[1,1]))/np.sin(mu)
        gamma = (1+alpha**2)/beta
        return (Q, beta, alpha, gamma)


def twiss2(beamline):
    '''Returns the Q, and the Twiss parameters beta, alpha, gamma of the beamline'''
    # 2x2 case
    OTM = getEquivalentElement(beamline)
    R = OTM[0]['matrix']
    
    if np.shape(R)[0]<4:
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
        mux = np.arccos(0.5*(Rx[0,0]+Rx[1,1]))
        if (Rx[0,1]<0): 
            mux = 2*np.pi-mux
        Qx = mux/(2*np.pi)
        betax = Rx[0,1]/np.sin(mux)
        alphax = (0.5*(Rx[0,0]-Rx[1,1]))/np.sin(mux)
        gammax = (1+alphax**2)/betax
        
        Ry = R[2:,2:]
        muy = np.arccos(0.5*(Ry[0,0]+Ry[1,1]))
        if (Ry[0,1]<0): 
            muy = 2*np.pi-muy
        Qy = muy/(2*np.pi)
        betay = Ry[0,1]/np.sin(muy)
        alphay = (0.5*(Ry[0,0]-Ry[1,1]))/np.sin(muy)
        gammay = (1+alphay**2)/betay
        
        return (Qx, betax, alphax, gammax, Qy, betay, alphay, gammay)


def computeTunes(f_f,f_d):
    fodo_lattice = Q4(f_f) + D4(l_drift) + B4(phi,l_dipole) + D4(l_drift) + Q4(f_d) + D4(l_drift) + B4(phi,l_dipole) + D4(l_drift)
    Qx, betax, alphax, gammax, Qy, betay, alphay, gammay = twiss(fodo_lattice)
    print(f'Qx = {Qx}')
    print(f'Qy = {Qy}')
    print(f'f_f = {f_f}')
    print(f'f_d = {f_d}')


def transportSigmas4D(sigma, beamline):
    coords = [sigma]
    s = [0]
    for element in beamline:
        coords.append(element['matrix'] @ coords[-1] @ element['matrix'].transpose())
        s.append(s[-1] + element['length']) 
    coords = np.array(coords)
    s = np.array(s)
    if len(sigma) < 4:
        return {'sigma11': coords[0][0],
                'sigma12': coords[0][1],
                'sigma21': coords[1][0], # equal to sigma12
                'sigma22': coords[1][1],
                's': s,
                'coords': coords,}
    elif len(sigma)==4:
        return {'sigma11': coords[0][0],
                'sigma12': coords[0][1],
                'sigma21': coords[1][0], # equal to sigma12
                'sigma22': coords[1][1],
                'sigma33': coords[2][2],
                'sigma34': coords[2][3],
                'sigma43': coords[3][2], # equal to sigma34
                'sigma44': coords[3][3],
                's': s,
                'coords': coords,}


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


# we need a minor update to the transportParticles
def transportParticles4D(x0,beamline):
    coords = [x0]
    s = [0]
    for elements in beamline:
        coords.append(elements['matrix'] @ coords[-1])
        s.append(s[-1] + elements['length']) 
    coords = np.array(coords).transpose()
    if len(x0)<4:
        return {'x':  coords[:,0,:],
            'xp': coords[:,1,:],
            's':  np.array(s),
            'coords': coords,}
    elif len(x0)==4:
        return {'x':  coords[:,0,:], # [particle_idx, s_idx]
            'xp': coords[:,1,:],     # [particle_idx, s_idx]
            'y':  coords[:,2,:],     # [particle_idx, s_idx]
            'py': coords[:,3,:],     # [particle_idx, s_idx]
            's':  np.array(s),       # [s_idx]
            'coords': coords,}       # [particle_idx, coord_idx, s_idx]


def generateParticles(nparticles, alpha, beta, stdgeoemittance):
    '''
    % [Xs, Angles] = generateParticles(nparticles, alpha, beta, stdgeoemittance)
    % generate nparticles accordingly to given Twiss parameters

    dgamba Sep 2014 (MATLAB)
    dgamba Sep 2022 (Python)
    '''
    # generate cov matrix
    myGamma = (1+alpha^2)/beta
    mySigma = np.array([[beta*stdgeoemittance, -alpha*stdgeoemittance],
                        [-alpha*stdgeoemittance, myGamma*stdgeoemittance]])

    # generate mean array
    myMean = np.array([0, 0])

    # actually generate particles and return coordinates
    (Xs, Angles) = np.random.multivariate_normal(myMean, mySigma, nparticles).T
    return (Xs, Angles)


def generateParticles2D(nparticles, alphaX, betaX, stdgeoemittanceX, alphaY, betaY, stdgeoemittanceY):
    ''' generateParticles2D(nparticles, alphaX, betaX, stdgeoemittanceX, alphaY, betaY, stdgeoemittanceY)
    It generates n particles distribuited accordingly to the given Twiss
    parameters with NO coupling between the two planes.
    
    dgamba Sep 2014 (MATLAB)
    dgamba Sep 2022 (Python)
    '''

    # generate sigma matrices
    myGammaX = (1+alphaX^2)/betaX
    mySigmaX = np.array([[betaX*stdgeoemittanceX, -alphaX*stdgeoemittanceX],
                        [-alphaX*stdgeoemittanceX, myGammaX*stdgeoemittanceX]])

    myGammaY = (1+alphaY^2)/betaY
    mySigmaY = np.array([[betaY*stdgeoemittanceY, -alphaY*stdgeoemittanceY],
                        [-alphaY*stdgeoemittanceY, myGammaY*stdgeoemittanceY]])

    # build sigma matrix
    mySigma = np.zeros(4,4)
    mySigma[0:1,0:1] = mySigmaX
    mySigma[2:3,2:3] = mySigmaY

    # build mean array
    myMean = np.array([0, 0, 0, 0])

    # generate particles
    (Xs, XAngles, Ys, YAngles) = np.random.multivariate_normal(myMean, mySigma, nparticles).T
    return (Xs, XAngles, Ys, YAngles)



###################################################
# Miscallenous functions
###################################################

def area(a, b, c) :
    '''
    this is the formula of the area of a triangle
    a,b,c are the phase-space coordinates of three particles
    '''
    return 0.5 * norm( np.cross( b-a, c-a ) )