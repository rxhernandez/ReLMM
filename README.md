Reinforcement Learning-based Material Models (ReLMM)
----------------

Materials discovery is a cornerstone of scientific 
and technological progress,
with far-reaching implications across various industries, 
including energy,
electronics, healthcare, and environmental sustainability.
However, often times it becomes difficult to understand which 
physical features
are most correlated to a given material property 
that we want to optimize.
Here, we introduced, ReLMM, a 
reinforcement learning-based feature engineering 
tool that finds an optimal physical features subset that can model a given target material property, 
for example the band gap for a semiconductor.
In our method, the feature subset is efficiently
learned through the self-playing framework
in reinforcement learning. 
Our results are demonstrated on hierarchical 
synthetic datasets and material datasets,
both of which show that ReLMM is able to 
find a minimal optimal dataset when compared to 
state-of-the-art feature selection tools such as 
LASSO and XGBoost.
Overall, an optimal set of features will enable 
better machine optimization
since we 'automatically learn' which 
features the machine should 'learn' from.

<hr>

Installation
----------------

ReLMM requires Python 3.11 or later to run.
We recommend running ReLMM in a python 
virtual environment or conda. ReLMM can
be installed together with other python
packages in the virtual environment or 
as a standalone package in its own 
virtual environment. We recommend the latter
to avoid any dependency conflicts with other
python packages for your project. Below is
an example of creating a python virtual
environment for ReLMM:

$ python3.11 -m venv ~/python_venvs/ReLMM

Once the virtual environment is created
and activated, you will have to build the 
ReLMM package from the source distribution (sdist). 
For this, first ensure you have the build python package.

$ pip install --upgrade build

Once the build package is installed, run 
the following command in the terminal.

$ python -m build

This command runs the build backend 
(in this case, setuptools), which copies all the 
sdist, gets the package dependencies and creates
a python wheel (.whl) file. The python wheel is 
essentially a zip file with a specially formatted
name and comes in a ready to install format with
pip. Running the below command should install
ReLMM in your virtual environment.

$ pip install dist/ReLMM-0.0.1-py3-none-any.whl

Code Testing and Confirmation
----------------

To test if ReLMM is installed correctly, 
please run the unittests included in 
the tests directory. To test the enviroment
module, for example, run:

$ python test_environment.py

Documentation
----------------

This repository contains code to implement ReLMM. It 
was created within the CONDA enviroment, and instructions 
for installing it are available in the 
[user guide](https://github.com/rxhernandez/ReLMM/blob/main/user_guide.md), 
though porting to other environnments (as long as the necessary
libraries are imported) should also be possible without additional
code.

* For details of the datasets and how we trained ReLMM 
please refer to the paper, noted in the "Citing" section below.

* Any questions or comments please reach out via email
to the authors of the paper.


<hr>

Authors
----------------

The ReLMM codes and databaess were developed by Maitreyee Sharma Priyadarshini, Nikhil K. Thota and Rigoberto Hernandez

Contributors can be found [here](https://github.com/rxhernandez/RELMM/graphs/contributors).

<hr>

Citing
----------------

If you use database or codes, please cite the following papers:

>M. Sharma Priyadarshini, N. K. Thota and R. Hernandez, “ReLMM: Reinforcement learning optimizes feature selection in material models"

>N. K. Thota, M. Sharma Priyadarshini and R. Hernandez, “NestedAE: Interpretable Nested Autoencoders for Multi-Scale Material Modelling,” _Mater. Horiz._, **11**, 700, (2024). [(0.1039/D3MH01484C)](http://doi.org/10.1039/D3MH01484C)

and/or this site:

>M. Sharma Priyadarshini, N. K. Thota and R. Hernandez, ReLMM, URL, [https://github.com/rxhernandez/RELMM](https://github.com/rxhernandez/RELMM)

<hr>

Acknowledgment
----------------

This work was supported by 
the Department of Energy (DOE), Office of Science, Basic Energy Science (BES), under Award #DE-SC0022305.


<hr>

License
----------------

ReLMM code and databases are distributed under terms of the [MIT License](https://github.com/rxhernandez/RELMM/blob/main/LICENSE).

