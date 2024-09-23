# Hands-On Lattice and Longitudinal Calculations - Python version
*D. Gamba, A. Latina, T. Prebibaj, A. Poyet, G. Russo, F. Soubelet, G. Sterbini, V. Ziemann*

During the [CAS 2024 in Santa Susanna](https://indico.cern.ch/event/1356988/) (Spain), we will be using Python as scripting languages for the Hands-On Lattice Calculation course.

This repository contains all material used during the the course.
The repository is based on the material prepared and updated during previous courses by *G. Sterbini, A. Latina, A. Poyet,* CERN and *V. Ziemann,* Uppsala University.

Before to start, please make sure you have a working Python installation. You can find setup instruction in a dedicated [repository](https://github.com/cerncas/hands-on-python/blob/main/Setup_Instructions.md). We kindly ask the student to read this document **before coming** to CAS to **prepare yourself** (and **your laptop**) for the course. 


- The `*.ipynb` notebooks will be the "canvas" used during the course. They contain all exercise with "incomplete" parts that the student will have to fill in. Solutions to all the exercises are also provided in `*_solutions.ipynb`. These notebooks are provided for the student convenience, but they are clearly not expected to be used before and/or during the course itself...
    - [01_Single_Particle](./01_Single_Particle_Optional_Physics.ipynb) Exercises to probe single particle dynamics.
    - [02_Multi_Particles](./02_Multi_Particles.ipynb) Exercises to probe multi-particle dynamics.
    - [03_Periodic_Systems](./03_Periodic_Systems.ipynb) Exercises to probe the concept of periodic lattices based on FODO cells.
    - [04_Dispersion](./04_Dispersion.ipynb) Exercises to introduce the simplest energy effect: dispersion.
    - [05_4D_Systems](./05_4D_Systems.ipynb) Exercises meant to extend the exploration to 4D systems (i.e. H and V).
- Additional `Optional*.ipynb` notebooks are also provided for who is fast and/or wants to explore deeper either some physics concepts or the phython implementation of our libraries.
- The `tracking_library*.py` files contain simple functions to ease implementing simple tracking of particle in Python.

> **NOTE:** the material in this repository is expected to evolve over time thanks to the feedback received from you! Please don't hesitate to transmit us your comments, suggestions, and complains!

### Known schools using this material

- [CAS 2024 in Santa Susanna](https://indico.cern.ch/event/1356988/)
- [CAS 2023 in Santa Susanna](https://indico.cern.ch/event/1226773/)
- [CAS 2022 in Kaunas](https://indico.cern.ch/event/1117526/)
- [CAS 2021 in Chavannes-des-Bogis](https://indico.cern.ch/event/1022988/)

## Notes for the maintainer and presenter

### Edit the material

The material is hosted on github under the [cerncas organisation](https://github.com/cerncas/) and mirrored (for backup purposes) to the CERN GitLab [CAS group](https://gitlab.cern.ch/cas).
One is expected to edit the material:

1. from **github directly** using the github editor
2. from **one's computer** cloning the repository, editing/adding/deleting the desired content, finally pushing the content to github

### (CERN) GitLab vs GitHub

See article [KB0003132](https://cern.service-now.com/service-portal?id=kb_article&n=KB0003132) to learn about CERN policy.
To setup a "Pull mirroring" on the CERN GitLab to retrieve a copy of GitHub repository, see the [official documentation](https://docs.gitlab.com/ee/user/project/repository/mirror/pull.html).

### Create a pdf of an .md file

The typically suggested way is to use `pandoc` package:

```bash
pandoc Setup_Instructions.md -o Setup_Instructions.pdf
```

unfortunately, this doesn't work when you have HTML inside your `.md` file, as we presently have...
A solution could be to use the [Print extension](https://marketplace.visualstudio.com/items?itemName=pdconsec.vscode-print) for VisualStudio...

### Update links to events

E.g. update all the links to CAS school event of the year using `sed`

```bash
sed -i -e 's/1117526\/contributions\/4978192/1356988\/contributions\/5713241/g' 01_Single_Particle_solutions.ipynb
```

### During the course

The students are expected to download the [whole repository](https://github.com/cerncas/hands-on-lattice-exercises/archive/refs/heads/master.zip) on their computer, and open the various notebooks using Jupyter Lab.
The presenter can also use Jupyter Lab and do the exercise with the students. To launch Jupyter Lab, move in a terminal to this folder and execute:

```bash
jupyter lab
```

One can also present [00_Introduction.ipynb](./00_Introduction.ipynb) in presentation mode:

```bash
jupyter nbconvert 00_Introduction.ipynb --to slides --post serve
```

Alternatively, one can:

- create a **html** of the slides:
   ```bash
   jupyter nbconvert 00_Introduction.ipynb --to slides
   ```
- create a **pdf** of the slides:
   ```bash
   conda install pandoc
   jupyter nbconvert 00_Introduction.ipynb --to pdf
   ```
