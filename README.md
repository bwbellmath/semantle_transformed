# Brian's Transformed Semantle Installation:
1. Download and install mamba by opening a Terminal and typing the
following:
`curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh`
and hitting `[Enter]`. 
Answer `yes` 2-4 times to various questions during installation. 

2. Install dependencies by typing `mamba install pytorch numpy pandas
   matplotlib sentence-transformers` into the terminal and hit
   `[Enter]` answer `Y` or `yes` when asked if you want to install the
   requisite packages. 

3. run semantle by typing `python semantle.py` and hitting `[Enter]`
   (It will take a bit longer the first time it runs because it needs
   to set up the word transformer, everything next will be faster)


