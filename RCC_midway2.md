## Deep-learning-practice using the RCC Midway2 cluster

### Grabbing a gpu2 node from midway2.rcc.uchicago.edu (after login)
```sh
sinteractive -p gpu2 --gres=gpu:1 --time=1:00:00
```

### Using pytorch + fastai
1. To use the pytorch + fastai library, see https://github.com/fastai/fastai and install fastai

2. If installed, Load the necessary modules to use pytorch + fastai
```sh
module load cuda/9.0
module load Anaconda3/5.0.1
conda activate fastai
```

3. To see if the pytorch works, download the pytorch examples and run. This should run the MNIST classification.
```sh
git clone https://github.com/pytorch/examples.git
cd pytorch_examples/mnist
python main.py
```

### Using jupyter notebook on the gpu2 node
For jupyter notebook related instructions, see https://git.rcc.uchicago.edu/ivy2/Jupyter_on_compute_nodes

1. Load Anaconda3 module
2. Find out an ip address of the node:
```sh
(fastai) [kywch@midway2-gpu01 ~]$ ifconfig eth0 | grep 'inet '
        inet 10.50.221.191  netmask 255.255.252.0  broadcast 10.50.223.255
```
3. Start jupyter without launching a browser on the node (add & at the end to run in the background), using the ip address obtained above: 
```sh
(fastai) [kywch@midway2-gpu01 ~]$ jupyter notebook --no-browser --ip=10.50.221.191 &
```
4. By default, it would listen on port 8888. However, if the port is already taken by another user, it will complain. Try the next available port with --port=<port number> option.

5. Eventually, you'll get a URL with a token. For example: http://10.50.221.191:8888/?token=2af56958386caf78d5bdc2086b20b6ff18553b701c581cd3

6. Point the browser running on your laptop/desktop to this URL. Notice: compute nodes are only visible on internal uchicago network. Therefore, you either need to be on campus or use VPN to be able to use this method.

7. **To use GPU-accelerated libraries in jupyter notebook, use Python [conda env:DL_GPU] kernel.** In the notebook, check whether it works by running
```sh
torch.cuda.is_available()   --> Should be True
torch.backends.cudnn.enabled    --> Should be True
```

8. Monitor the gpu usage by running nvidia-smi. The below script will update the gpu usage every second. Use ctrl-c to stop it.
```sh
(fastai) [kywch@midway2-gpu01 ~]$ nvidia-smi -l 1
```

9. To kill jupyter, ps to get the pid and use kill command.




