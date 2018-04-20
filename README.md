# deep-learning-practice
Using the RCC Midway2 cluster

Login to the midway2.rcc.uchicago.edu and grab a gpu node
```sh
sinteractive -p gpu2 --gres=gpu:1 --time=1:00:00
```

To use the pytorch + fastai library, see https://github.com/fastai/fastai and install fastai

Load the necessary modules to use pytorch + fastai
```sh
module load cuda/9.0
module load Anaconda3/5.0.1
conda activate fastai
```

To see if the pytorch works, download the pytorch examples and run. This should run the MNIST classification.
```sh
git clone https://github.com/pytorch/examples.git
cd pytorch_examples/mnist
python main.py
```




