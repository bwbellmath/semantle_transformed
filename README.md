# Brian's Jerry-Rigged Transformed Semantle Installation:
1. Download the source code for Semantle from Git by clicking `<>
   Code` and selecting "Download ZIP" from the drop-down. Open your
   downloads folder and extract the downloaded files to a folder `semantle_transformed`
2. Open a terminal and type `cd Downloads` to navigate to the folder
   to which you saved the zip file with the code
3. Download and install mamba by opening a Terminal and pasting the
following:
`curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh`
and hitting `[Enter]`. 
Answer `yes` 2-4 times to various questions during installation. 

4. Install dependencies by pasting `mamba install pytorch numpy pandas
   matplotlib sentence-transformers` into the terminal and hit
   `[Enter]` answer `Y` or `yes` when asked if you want to install the
   requisite packages. 

5. run semantle by typing `python semantle.py` and hitting `[Enter]`
   (It will take a bit longer the first time it runs because it needs
   to set up the word transformer, everything next will be faster)


