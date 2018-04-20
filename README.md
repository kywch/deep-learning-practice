# deep-learning-practice
Using the RCC Midway2 cluster

Login to the midway2.rcc.uchicago.edu and grab a gpu node
```sh
sinteractive -p gpu2 --gres=gpu:1 --time=1:00:00
```

Load the necessary modules to use pytorch
```sh
module load cuda/9.0
module load Anaconda3/5.0.1
conda activate pytorch_cuda_9.0
```

To use the fastai library, see https://github.com/fastai/fastai

