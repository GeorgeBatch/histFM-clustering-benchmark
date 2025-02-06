```shell
# create conda environment
conda create -n tiatoolbox-simple python=3.11 -y
conda activate tiatoolbox-simple

# install prerequisites and tiatoolbox
conda install openslide openjpeg -c conda-forge # openslide - C library

# tiatoolbox >= 1.6.0
pip install tiatoolbox>=1.6.0
# if tiatoolbox is already installed and you want to upgrade
# pip install --ignore-installed --upgrade tiatoolbox

# used in notebooks, but not installed with tiatoolbox
conda install seaborm -c conda-forge
```
