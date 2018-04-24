## deep-learning-practice
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

For jupyter notebook related instructions, see https://git.rcc.uchicago.edu/ivy2/Jupyter_on_compute_nodes

To use jupyter notebook on the gpu node, load Anaconda3 module and
1. Find out an ip address of the node:
```sh
(fastai) [kywch@midway2-gpu01 ~]$ ifconfig eth0 | grep 'inet '
        inet 10.50.221.191  netmask 255.255.252.0  broadcast 10.50.223.255
```
2. Start jupyter without launching a browser on the node, using the ip address optained above: 
```sh
(fastai) [kywch@midway2-gpu01 ~]$ jupyter notebook --no-browser --ip=10.50.221.191
```
3. By default, it would listen on port 8888. However, if the port is already taken by another user, it will complain. Try the next available port with --port=<port number> option.
4. Eventually, you'll get a URL with a token. For example: http://10.50.221.191:8888/?token=2af56958386caf78d5bdc2086b20b6ff18553b701c581cd3
5. Point the browser running on your laptop/desktop to this URL. Notice: compute nodes are only visible on internal uchicago network. Therefore, you either need to be on campus or use VPN to be able to use this method.
6. **To use GPU-accelerated libraries in jupyter notebook, use Python [conda env:DL_GPU] kernel.**
7. To kill jupyter, press Ctrl+c and then confirm with y that you want to stop it




