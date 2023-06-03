# Brian's Transformed Semantle Installation:
1. Download and install mamba [mamba
   forge]{https://github.com/conda-forge/miniforge#mambaforge}
2. run `curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh`
3. Type `mamba install pytorch numpy pandas matplotlib sentence-transformers` into the terminal and hit `[Enter]`
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
import os
import argparse
