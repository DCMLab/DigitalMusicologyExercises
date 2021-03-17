# Digital Musicology Exercises

## Getting Started

1. Install [git](https://git-scm.com/) or the [GitHub desktop app](https://desktop.github.com/).
2. Clone this repository.
3. Setup `python` and `jupyter`.
   - If you have not used python and/or jupyter before, see below for installation instructions.
4. Install `pandas` and `cufflinks` (again, see below)
5. Run `jupyter` and open the notebook `tone_profiles.ipynb`

## Installing Python

If you have not installed python before or don't know how to get started, follow these steps:

1. [Install `conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
   Go to "Regular Installation" and select your operating system.
   We advise to install Miniconda instead of Anaconda, as it does not require download a large file.
2. Start conda (Windows) or a terminal (Mac/Linux)
3. Create a new environment using
   ```
   conda create --name pitch_profiles python=3.8

   ```
   Environments are useful to install dependencies for different projects separately.
4. Activate your new environment using
   ```
   conda activate pitch_profiles
   ```
5. Once the `pitch_profiles` environment is active, run the following commands:
   ```
   pip install -U pip
   pip install -U jupyter pandas cufflinks
   python -m ipykernel install --user --name pitch_profiles

   ```
6. Run jupyter using
   ```
   jupyter notebook
   ```
   You browser should open a new tab with jupyter.
   Navigate to the repo that you cloned before and start the notebook `tone_profiles.ipynb`.
7. Enjoy!
