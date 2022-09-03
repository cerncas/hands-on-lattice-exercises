# Hands-On Lattice and Longitudinal Calculations - Setup Instructions
---

During the course we will use **Python3** in a **Jupyter notebook** with [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) and, mostly, the [numpy](https://numpy.org/) and [matplotlib](https://matplotlib.org/) packages. We will explain in the following sections how to install all necessary software on **your laptop**.
A basic knowledge of Python is assumed. If you are not familiar with Python, you can find a few resources to fill the gap in the following sections.

To get a better idea of the level of the Python knowledge needed for the course you can browse the [primer of the hands-on course](CAS_Optics_Primer.pdf). Do not worry about the theory for the moment (it will be discussed in details during the school) but focus on the Python syntax and data types (tuples, lists,...).

After [a short introduction](#a-very-short-introduction-to-python), where we provided some useful links to get familiar with Python, we will focus on the [software setup](#software-setup). 
Finally, in [appendix](#appendix-python-packages) you will find links and cheatsheets for the most common Python packages that will be used during the course.

> **Important:** we kindly ask you to go throw this document **before coming** to CAS, such as to **prepare yourself** (and **your laptop**) for the course. 

---
# A very short introduction to Python
You can find several nice courses, videos and resources on the internet. Here you have a couple of suggestions.

[![Python for Beginners - Learn Python in 1 Hour](http://img.youtube.com/vi/kqtD5dpn9C8/0.jpg)](http://www.youtube.com/watch?v=kqtD5dpn9C8)

[![Learn Python - Full Course for Beginners](http://img.youtube.com/vi/rfscVS0vtbw/0.jpg)](http://www.youtube.com/watch?v=rfscVS0vtbw)

### Test Python on a web page

If you are not familiar with Python and you have not it installed on your laptop, you can start playing with simple python snippets on the web. Without installing any special software you can connect, 
for example, to [jupyterLab](https://gke.mybinder.org/v2/gh/jupyterlab/jupyterlab-demo/try.jupyter.org?urlpath=lab),
and test the following commands:

```python=
import numpy as np

# Matrix definition
Omega=np.array([[0, 1],[-1,0]])
M=np.array([[1, 0],[1,1]])

# Sum and multiplication of matrices
Omega - M.T @ Omega @ M
# M.T means the "traspose of M".

# Function definition
def Q(f=1):
    return np.array([[1, 0],[-1/f,1]])

# Eigenvalues and eigenvectors
np.linalg.eig(M)
```
You can compare and check your output with the ones [here](tests/SimpleTest.ipynb).

---
# Software Setup

In this section we will explain how to install Python and JupyterLab on your laptop.
JupyterLab is a user-friendly environment to work with Python. 
You can find an overview on JupyterLab [here](https://jupyterlab.readthedocs.io/en/stable/).

## Installation

Please install the Anaconda distribution from
https://www.anaconda.com/distribution/

<p align="center">
<a href=https://www.anaconda.com/distribution/><img src="_img_instructions/anaconda.png" alt="" width="70%"/></a>
</p>

> We suggest to install one of the latest distribution (**for example version 3.9**).

Please, test the following code to check that all packages are correctly installed:

1. Launch Jupyter Lab from a terminal:

    ```bash
    jupyter lab
    ```

2. You should end-up on your default browser with a page similar to the following:
    <p align="center">
    <img src="_img_instructions/upload_5b0618b75e4f4df0facf2a609b9354b5.png" alt="" width="70%"/>
    </p>

    If you are not familiar with Python, you can start playing with simple python snippets.
    For example, have a look to the following [notebook](tests/PythonExample.ipynb) (courtesy of *Simon Albright*), and/or follow the instructions below.

## Test that everything works!

> Please, make sure to go throw all the examples below to familiarise with the typical Python concepts that will be used during the course, but also to verify your installation. 

### Indexing

Generate a random array and select specific elements:

```python=
import numpy as np

# Create an array
array1d = np.random.uniform(size=10)
# Print selected elements
print("Entire array: " + str(array1d) + "\n")
print("Specific element: " + str(array1d[5]) + "\n")
print("Last element: " + str(array1d[-1]) + "\n")
print("Specific elements: " + str(array1d[3:7]) + "\n")
print("First 5 elements array: " + str(array1d[:5]) + "\n")
print("Last 5 elements array: " + str(array1d[5:]) + "\n")
```
will result in, e.g.:

```bash
Entire array: [0.09402447 0.05647033 0.79670378 0.60573004 0.81588777 0.97863634 0.51376609 0.19763518 0.7649532  0.59285346]

Specific element: 0.9786363385079204

Last element: 0.5928534616865488

Specific elements: [0.60573004 0.81588777 0.97863634 0.51376609]

First 5 elements array: [0.09402447 0.05647033 0.79670378 0.60573004 0.81588777]

Last 5 elements array: [0.97863634 0.51376609 0.19763518 0.7649532  0.59285346]
```

### Implicit loops

In contrast to programming languages like C++, Python can handle vectors. No loop is required, e.g. to multiply each element with a constant or squaring it:

```python=
import numpy as np

# Create an array
array1d = np.random.uniform(size=10)

print("Entire array: " + str(array1d) + "\n")
print("Each element muliplied by 5: " + str(5 * array1d) + "\n")
print("Each element squared: " + str(array1d**2) + "\n")
print("Square root of each element: " + str(np.sqrt(array1d)) + "\n")
```
will result in, e.g.:

```bash
Entire array: [0.2240143  0.35153156 0.68864907 0.14062298 0.77280195 0.26872206 0.9135403  0.8776261 0.26158576 0.93883652]

Each element muliplied by 5: [1.12007151 1.75765782 3.44324537 0.70311488 3.86400975 1.34361029 4.56770149 4.3881305 1.30792879 4.69418259]

Each element squared: [0.05018241 0.12357444 0.47423755 0.01977482 0.59722285 0.07221154 0.83455588 0.77022757 0.06842711 0.88141401]

Square root of each element: [0.47330149 0.59290097 0.82984883 0.3749973  0.87909155 0.51838408 0.95579302 0.936817 0.51145455 0.96893577]
```


You can perform also some linear algebra tests:

```python=
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sympy as sy

# Matrix definition
Omega=np.array([[0, 1],[-1,0]])
M=np.array([[1, 0],[1,1]])

# Sum and multiplication of matrices
Omega - M.T @ Omega @ M
# M.T means the "traspose of M".

# Function definition
def Q(f=1):
    return np.array([[1, 0],[-1/f,1]])

#Eigenvalues and eigenvectors
np.linalg.eig(M)
```

### Plotting
You can create a simple plot as follow:
```python=
# check a simple plot
%matplotlib inline

plt.plot([0,10],[0,10],'ob-')
plt.xlabel('My x-label [arb. units]')
plt.ylabel('My y-label [arb. units]')
plt.title('My title')
```
which should result in the following plot:

<p align="center">
<img src="_img_instructions/upload_b7610a37c41729a79bc4b0a5d863594b.png" alt="" width="40%"/>
</p>

Or something fancier:
```python=
# an sns plot
sns.set(style="ticks")
rs = np.random.RandomState(11)
x = rs.normal(size=1000)
y = rs.normal(size=1000)
sns.jointplot(x=x, y=y, kind="hex")
```
<p align="center">
<img src="_img_instructions/upload_1c7eaa74ee5422c62408cc9a57f7f0de.png" alt="" width="40%"/>
</p>


### Pandas dataframes

You can import some information in a pandas dataframe from the internet :
```python=
# a simple pandas dataframe (GDP world statistics)
myDF=pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/.GDP.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en')
myDF[(myDF['TIME']==2018) & (myDF['MEASURE']=='MLN_USD')]
myDF.head()
```
that gives
<p align="center">
<img src="_img_instructions/upload_90f1a933f3f7092f2a4d4b50725e61ae.png" alt="" width="70%"/>
</p>

### Animations

> **IMPORTANT**: we will use animation in Python. Please check that the following code is running or your machine.

```python=
# to have the animation you need to configure properly your jupyter lab
# From https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6

# > pip install ipywidgets
# > jupyter nbextension enable --py widgetsnbextension
# > jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Possibly you need also nodejs 
# https://anaconda.org/conda-forge/nodejs
# > conda install -c conda-forge nodejs 

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive

t = np.linspace(0,10,1000)
def plotIt(f):
    plt.plot(t, np.sin(2*np.pi*f*t))
    plt.grid(True)
    
interactive_plot = interactive(plotIt,f=(0,1,.1),continuous_update=True)
output = interactive_plot.children[-1]
output.layout.height = '300px'
interactive_plot
```

<p align="center">
<img src="_img_instructions/upload_0dff4499bd5e7b21942e1990cd76d0e9.png" alt="" width="70%"/>
</p>

---
## Appendix: Python Packages

You can leverage python's capability by exploring a galaxy of packages. Below you can find the most useful for our course (focus mostly on `numpy` and `matplotlib`) and some very popular ones. 

### The *numpy* package
To get familiar with the *numpy* package have a look at the following [summary poster](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf).
You can google many other resources, but the one presented of the poster covers the set of instructions you should familiar with.
[![numpy cheat sheet](_img_instructions/upload_6ffb4d07b1ebb895528f2a34aae41ec6.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)

### The *matplotlib* package
To get familiar with the *matplotlib* package have a look at the following [summary poster](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf).
[![matplotlib cheat sheet](_img_instructions/upload_4b54812812e21978b600b860ba1ddf5b.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)

### The *linalg* module
To get familiar with the Linear Algebra (linalg) module have a look at the following [summary poster](
https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_SciPy_Cheat_Sheet_Linear_Algebra.pdf).
[![linalg cheat sheet](_img_instructions/upload_15561fc12184bb0ae3f9cf7b1850317a.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_SciPy_Cheat_Sheet_Linear_Algebra.pdf)

### The *pandas* package (optional)
To get familiar with the *pandas* package have a look at the following [summary poster](
https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf).
[![pandas cheat sheet](_img_instructions/upload_90383c01e29d29fb6a5516c613e22c4d.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)

### The *seaborn* package (optional)
To get familiar with the *seaborn* package have a look at the following [summary poster](
https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf).
[![seaborn cheat sheet](_img_instructions/upload_9a3c3f5ca48bbd567a0662df20dbd16f.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf)

### The *sympy* package (optional)
To get familiar with the *sympy* package have a look at the following [summary poster](http://daabzlatex.s3.amazonaws.com/9065616cce623384fe5394eddfea4c52.pdf).
[![sympy cheat sheet](_img_instructions/upload_fc7a06ea6135d2bf17311bd7a91f1a9f.png)](http://daabzlatex.s3.amazonaws.com/9065616cce623384fe5394eddfea4c52.pdf)