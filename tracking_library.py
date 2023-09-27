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

# ignore "FutureWarning"... (temporary patch for seaborn package issues...)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
