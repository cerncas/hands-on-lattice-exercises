# coding: utf-8
"""
CAS course on "Introduction to Accelerator Physics", 21 September - 04 October 2025, Santa Susanna, Spain, 2025.

This python package contains all support functions for the exercises of the "Hands-on on Lattice Calculations in Python" course.

Authors: D. Gamba, A. Latina, T. Prebibaj, F. Soubelet
"""

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
from ipywidgets import interactive

# some setup for the plotting
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (15, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
plt.rcParams.update(params)

# ignore "FutureWarning"... (temporary patch for seaborn package issues...)
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)


def D(L):
    """
    Drift space element of length L (2x2 matrix).

    Parameters
    ----------
    L : float
        Length of the drift [m]

    Returns
    -------
    element : list of dict
        A list with a single dictionary containing the matrix and length.
    """
    return [{'matrix': np.array([[1, L], [0, 1]]), 'length': L}]


def Q(f):
    """
    Thin-lens quadrupole element with focal length f (2x2 matrix).

    Parameters
    ----------
    f : float
        Focal length of the quadrupole [m]

    Returns
    -------
    element : list of dict
        A list with a single dictionary containing the matrix and length.
    """
    return [{'matrix': np.array([[1, 0], [-1/f, 1]]), 'length': 0}]


def B(phi, L):
    """
    Thick sector bend with deflecting angle phi and length L (2x2 matrix).

    Parameters
    ----------
    phi : float
        Deflection angle [rad]
    L : float
        Length of the bend [m]

    Returns
    -------
    element : list of dict
        A list with a single dictionary containing the matrix and length.
    """
    bend_matrix = np.array([[np.cos(phi), L/phi*np.sin(phi)],
                            [-phi/L*np.sin(phi), np.cos(phi)]])
    return [{'matrix': bend_matrix, 'length': L}]


def getEquivalentElement(beamline):
    """
    Compute the equivalent transfer matrix and total length of a beamline (list of elements).

    Parameters
    ----------
    beamline : list of dict
        List of elements each with a transfer matrix and length.

    Returns
    -------
    equivalent : list of dict
        A single element equivalent to the entire beamline.
    """
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
        length += elements['length']
    # we return the dict with the "usual" keys (matrix and length) embedded in a
    #  list (with a single element), as for the definition of the D and Q functions
    return [{'matrix': equivalentElement, 'length': length}]


def transportParticles(X_0, beamline, s_0=0):
    """
    Track particle(s) through a beamline.

    Parameters
    ----------
    X_0 : ndarray
        Initial particle coordinates [2 x M] (x and x' for M particles).
    beamline : list of dict
        Sequence of beamline elements.
    s_0 : float, optional
        Initial longitudinal position [m], by default 0.

    Returns
    -------
    result : dict
        Dictionary with keys:
            'x'      : [N x M] horizontal positions [s_index, particle_index],
            'xp'     : [N x M] angles [s_index, particle_index],
            's'      : [N] longitudinal positions [s_index],
            'coords' : [N x 2 x M] full coordinates [s_index, coord_index, particle_index].
            
    Disclaimer
    -------
    If beamline is made of 5 elements, the output will have 5+1 "elements" as it will also return include the initial particle coordinates.  
    """
    coords = [X_0]
    s = [s_0]
    for element in beamline:
        coords.append(element['matrix'] @ coords[-1])
        s.append(s[-1] + element['length']) 
    coords = np.array(coords)
    s = np.array(s)
    return {
            'x': coords[:, 0, :], # [s_index, particle_index]
            'xp': coords[:, 1, :], # [s_index, particle_index]
            's': s, # [s_index]
            'coords': coords # [s_index, coord_index, particle_index]
           }


def transportSigmas(sigma_0, beamline):
    """
    Transport a sigma matrix through a beamline.

    Parameters
    ----------
    sigma_0 : ndarray
        Initial 2x2 sigma matrix.
    beamline : list of dict
        Sequence of beamline elements.

    Returns
    -------
    result : dict
        Dictionary with keys:
            'sigma11', 'sigma12', 'sigma21', 'sigma22' : [N] sigma matrix elements,
            's'                                        : [N] longitudinal positions,
            'sigmas'                                   : [N x 2 x 2] full sigma matrices.

    Disclaimer
    -------
    If beamline is made of 5 elements, the output will have 5+1 "elements" as it will also return include the initial particle coordinates.  
    """
    sigmas = [sigma_0]
    s = [0]
    for element in beamline:
        sigmas.append(element['matrix'] @ sigmas[-1] @ element['matrix'].T)
        s.append(s[-1] + element['length']) 
    sigmas = np.array(sigmas)
    s = np.array(s)
    return {'sigma11': sigmas[:, 0, 0],
            'sigma12': sigmas[:, 0, 1],
            'sigma21': sigmas[:, 1, 0],
            'sigma22': sigmas[:, 1, 1],
            's': s,
            'sigmas': sigmas}


def twiss(beamline):
    """
    Compute Twiss parameters and tune for a periodic beamline.

    Parameters
    ----------
    beamline : list of dict
        Sequence of beamline elements.

    Returns
    -------
    tune : float
        Betatron tune (fraction of oscillation per turn).
    beta : float
        Beta function at entrance [m].
    alpha : float
        Alpha function at entrance.
    gamma : float
        Gamma function at entrance.
    """
    
    # first, compute the equivalent "One-Turn-Map", and extract its matrix:
    OTM = getEquivalentElement(beamline)
    R = OTM[0]['matrix']
    
    # check that this matrix is stable:
    if np.abs(0.5 * (R[0, 0] + R[1, 1])) > 1:
        raise ValueError('This beamline is not stable!')
    
    # all relevant Twiss parameters can be extrcted from the matrix:
    mu = np.arccos(0.5 * (R[0, 0] + R[1, 1]))
    if R[0, 1] < 0:
        mu = 2 * np.pi - mu

    tune = mu / (2 * np.pi)
    beta = R[0, 1] / np.sin(mu)
    alpha = 0.5 * (R[0, 0] - R[1, 1]) / np.sin(mu)
    gamma = (1 + alpha ** 2) / beta

    return tune, beta, alpha, gamma


def D3(L):
    """
    Drift space element of length L (3x3 matrix to account for energy).

    Parameters
    ----------
    L : float
        Length of the drift space [m]

    Returns
    -------
    list of dict
        List containing a dictionary with the transfer matrix and its length.
    """
    matrix = np.array([[1, L, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    return [{'matrix': matrix, 'length': L}]


def Q3(f):
    """
    Thin-lens quadrupole element with focal length f (3x3 matrix to account for energy).

    Parameters
    ----------
    f : float
        Focal length of the quadrupole [m]

    Returns
    -------
    list of dict
        List containing a dictionary with the transfer matrix and zero length.
    """
    matrix = np.array([[1, 0, 0],
                       [-1/f, 1, 0],
                       [0, 0, 1]])
    return [{'matrix': matrix, 'length': 0}]


def B3(phi, l):
    """
    Thick sector bend with deflecting angle `phi` and length `L` (3x3 matrix to account for energy).

    Parameters
    ----------
    phi : float
        Bending angle [rad]
    l : float
        Arc length of the bend [m]

    Returns
    -------
    list of dict
        List containing a dictionary with the transfer matrix and its length.
    """
    matrix = np.array([
        [np.cos(phi), l/phi*np.sin(phi), l/phi*(1 - np.cos(phi))],
        [-phi/l*np.sin(phi), np.cos(phi), np.sin(phi)],
        [0, 0, 1]
    ])
    return [{'matrix': matrix, 'length': l}]


__packages = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    # "ipympl": "ipympl",
}
__setup_ok = True
for __package, __import_name in __packages.items():
    try:
        __module = __import__(__import_name)
        print(f"{__package} is installed, version: {__module.__version__}")
    except ImportError:
        print(f"{__package} is not installed")
        __setup_ok = False

if __setup_ok:
    print("-> Setup is OK! Have fun!")
else:
    print("-> Setup is NOT OK!")