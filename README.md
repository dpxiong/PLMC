# PLMC
PLMC is a deep learning framework for protein crystallization prediction with protein language embeddings and handcrafted features.

## Creating a conda environment
After cloning the repository, please run the following lines to install an environment for PLMC.
```
conda create -n plmc python=3.9
conda activate plmc
conda install numpy=1.23.5
conda install -c pytorch pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
conda install -c conda-forge scikit-learn=1.0.2
pip install fair-esm
```

## Data preparation
Download the data at this [address](https://zenodo.org/record/6475529/), and uncompress it to the current directory.
The protein language embeddings can be generated for each dataset as below:

```
python esm2_650m.py CRYS_DS #take CRYS_DS as an example
```

## Running PLMC
Execute the following command:
```
python main.py --rawpath ./data/CRYS_DS
```

If you have any questions, please contact Dapeng Xiong at dpxiong@live.com.
