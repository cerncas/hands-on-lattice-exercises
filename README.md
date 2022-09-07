# Hands-On Lattice and Longitudinal Calculations - Python version
*D. Gamba, A. Latina, A. Poyet, G. Sterbini, V. Ziemann*

During the [CAS 2022 in Kaunas](https://indico.cern.ch/event/1117526/) (Lithuania), we will be using Python or MATLAB as scripting languages for the Hands-On Lattice Calculation course.

This [repository](https://github.com/dgamba/CAS_2022) contains all material used during the **python** version of the course.
The repository is based on the material prepared and updated during previous courses by *G. Sterbini, A. Latina, A. Poyet,* CERN and *V. Ziemann,* Uppsala University.

- [Setup_Instructions.md](./Setup_Instructions.md) contains all information to get ready for the course. We kindly ask the student to read this document **before coming** to CAS to **prepare yourself** (and **your laptop**) for the course. 
- [CAS_Optics_Primer.pdf](./CAS_Optics_Primer.pdf) is the primer of the course. You are invited to consult it before, during, and after the course. 
- [Exercises.ipynb](./Exercises.ipynb) will be the "canvas" used during the course. It contains all exercise with "incomplete" parts that the student will have to fill.
- [Exercises_Solutions.ipynb](./Exercises_Solutions.ipynb) contains the solutions to all the exercises that we will tackle during the course. This notebook is provided for the student convenience, but it is clearly not expected to be used before and/or during the course itself.

> **NOTE:** the material in this repository is expected to evolve over time thanks to the feedback received from you! Please don't hesitate to transmit us your comments, suggestions, and complains!

### Known schools using this material

- [CAS 2022 in Kaunas](https://indico.cern.ch/event/1117526/)
- [CAS 2021 in Chavannes-des-Bogis](https://indico.cern.ch/event/1022988/)


## Notes for the maintainer and presenter

### Edit the material

The material is hosted on github under the [cerncas organisation](https://github.com/cerncas/) and mirrored (for backup purposes) to the CERN GitLab [CAS group](https://gitlab.cern.ch/cas).
One is expected to edit the material:

1. from **github directly** using the github editor
2. from **one's computer** cloning the repository, editing/adding/deleting the desired content, finally pushing the content to github

### Create a pdf of an .md file

The typically suggested way is to use `pandoc` package:

```bash
pandoc Setup_Instructions.md -o Setup_Instructions.pdf
```

unfortunately, this doesn't work when you have HTML inside your `.md` file, as we presently have...
A solution could be to use the [Print extension](https://marketplace.visualstudio.com/items?itemName=pdconsec.vscode-print) for VisualStudio...

### During the course

The students are expected to download [Exercises.ipynb](./Exercises.ipynb) on their computer, and open it using Jupyter Lab.
The presenter can also use Jupyter Lab and do the exercise with the students, or can present [Exercises_Solutions.ipynb](./Exercises_Solutions.ipynb) in presentation mode:

```bash
jupyter nbconvert Exercises_Solutions.ipynb --to slides --post serve
```

Alternatively, one can:

- create a **html** of the slides:
   ```bash
   jupyter nbconvert Untitled2.ipynb --to slides
   ```
- create a **pdf** of the slides:
   ```bash
   jupyter nbconvert Untitled2.ipynb --to pdf
   ```