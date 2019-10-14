### Download to data directory

https://s3.eu-central-1.amazonaws.com/i008data/fashion-mnist_train.csv  
https://s3.eu-central-1.amazonaws.com/i008data/fashion-mnist_test.csv  
https://s3.eu-central-1.amazonaws.com/i008data/small_sneaker_not_sneaker.tar.gz


### Environment
```bash
conda env update
conda install nb_conda
source activate dl
python -m ipykernel install --user --name myenv --display-name "dsr-dl"

If you dont have a GPU in your system remove the cudatoolkit from the environment.yml file

```


### Commands

```bash
tensorboard --port=8888 --logdir runs
```
