# Hands-On Lattice and Longitudinal Calculations - Presentation Instructions
---

> Those instructions are meant for the be used by the presenter only

## Edit the material

The material is hosted on github and mirrored to the CERN gitlab.
One is expected to edit the material:

1. from **github directly** using the github editor
2. from **one's computer** cloning the repository, editing/adding/deleting the desired content, finally pushing the content to github
3. from overleaf. The project xxxx (read_only) is also synched (manually) with github repository



## During the course

The students are expected to download and open the [Exercises.ipynb](./Exercises.ipynb) on their computer using Jupyter Lab.
The presenter is also expected to use Jupyter Lab and do the exercise with the students.

At the same time, the presenter can present [Exercises_Solutions.ipynb](./Exercises_Solutions.ipynb) in presentation mode:

```bash
jupyter nbconvert Exercises_Solutions.ipynb --to slides --post serve
```

Alternatively, one can:

- crate a **html** of the slides:
   ```bash
   jupyter nbconvert Untitled2.ipynb --to slides
   ```
- crate a **pdf** of the slides:
   ```bash
   jupyter nbconvert Untitled2.ipynb --to pdf
   ```