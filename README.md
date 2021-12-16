# Meta_AMP
Meta learning approach to discovering new Antimicrobial peptide candidates



## Instructions


The configs are specified in `config/`. Configs are by default `DEFAULT_CONFIG` in `config/default.py`. You can override the default configs by specifying them in `config/AMP_configs.py`, `config/debug_configs.py`, etc... After overriding, you can run the experiments with the name of the config -- for example:



### Instructions for setup on Compute Canada (not RNA14)

You can create a python virtual environment and run `pip install -r requirements.txt`. Afterwards, it should work.


### Instructions for setting up RNA14 on the Mila Cluster

Unfortunately, Compute Canada doesn't support anaconda, so you will need to run on the Mila Cluster

Run the following:
```
module load anaconda/3
conda create --name metarlbo_env python=3.7
conda activate metarlbo_env
conda install -c bioconda viennarna
```

Next, try to run `pip install -r requirements.txt`.

You will likely run into issues with some packages -- for those packages, you need to install via `conda`. Here are some examples:
```
conda install cloudpickle
conda install urllib3
```

After fixing those `pip install -r requirements.txt` should work. Alternatively, you can try to install the packages individually and if there are any errors with the existing packages, you can run `conda install packagename`.



## Running Experiments (Examples)

```
python main.py debug_KNR

python main.py debug_RNA14

python main.py amp_knr_016
```


## View Logs

To view tensorboard logs:
```
tensorboard --logdir logs
```




## Dependency notes (Not relevant for now)
- For blast make sure to install ncbi-blast+ locally, on CC it's already installed you just need to load the following modules:
module load nixpkgs/16.09 gcc/7.3.0 blast+/2.9.0







<!-- 

conda install -c bioconda viennarna



Setup README.md
Setup -- Tensorflow== 2.3

 -->