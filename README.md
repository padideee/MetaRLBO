# Meta_AMP
Meta learning approach to discovering new Antimicrobial peptide candidates

After cloning the repository, run:
```
cd Meta_AMP
pip install -e . 
```
to install Meta-AMP package.



## Instructions


The configs are specified in `config/`. Configs are by default `DEFAULT_CONFIG` in `config/default.py`. You can override the default configs by specifying them in `config/AMP_configs.py` or `config/debug_configs.py`. After overriding, you can run the experiments with the name of the config -- for example:

```
python main.py debug_KNR

python main.py amp_knr_016
```


## View Logs

To view tensorboard logs:
```
cd logs
tensorboard --logdir .
```




