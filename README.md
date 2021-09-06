# Beam project, TU Berlin, SS2021
### Sergi Andreu, Dylan Everingham, Carsten van de Kamp, Sebastian Myrbäck
Project course in numerical analysis, on the topic of FEM simulation of the bending of beams using Bernoulli theory.

### Structure

* The full report pdf is found [here](Report_Numerical_Analysis_of_Vibrating_Beams.pdf). 
* **Attachment A** from the report contains the code. The code is found here in the repository, and the simulations are found in notebooks. Which notebooks that correspond to which parts in the report is presented below.

  1. **Static case**: The notebook for these simulations is in [notebook.ipynb](notebook.ipynb). The methods [meshes.py](meshes.py), [AnalyticalSolutions.py](AnalyticalSolutions.py) and [NumericalSolutions.py](NumericalSolutions.py) are used in this notebook. 
  2. **Dynamic case**: The simulations for the *Newmark method* are performed in [Dynamic_Case.ipynb](Dynamic_Case.ipynb) and the simulations for the *eigenvalue method* are performed in [eigenvalue_method.ipynb](eigenvalue_method.ipynb). Here, additionally, the methods [DynamicSolutions.py](DynamicSolutions.py) and [Eigenvalues.py](Eigenvalues.py) are used respectively.
* **Attachment B** from the report contains the animations. This can be found under in Report folder.


