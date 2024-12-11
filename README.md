# SEPC

This repository contains the official code for the paper "Structural Entropy Guided Probabilistic Coding".

## Quick start
1. Download the pretrained model into SEPC/ptms/

2. Install dependencies
``` bash
pip install -r requirements.txt
```

3. Run examples
``` bash
bash scripts/run_offensive.sh
```

Parameters for other datasets could be found in `wandb_script_roberta`

## Use wandb to sweep your best parameters

1. Define the parameter combinations in wandb_script_roberta

2. Create a sweep controller
``` bash
wandb sweep wandb_script_roberta/tweeteval_offensive.yaml
```

3. Run sweep agent
``` bash 
wandb agent ***
```


## Credits
The code and datasets in this repository are based on [SPC](https://github.com/zerohd4869/SPC).
