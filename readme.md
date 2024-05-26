# Over-Smoothing
Verify the over-smoothing phenomenon in common neural networks

## How to use ?

The trainnning config files is in the main folders.

You can modify the model architecture and select different dataset for node classification tasks.

Then use 

```cmd

python  train.py -c your_config_file.py --gpus your_gpu_numbers

```

Whole training architecture is written in [easy-torch](https://github.com/cnstark/easytorch)

Training results is in 'experiments' folders.
