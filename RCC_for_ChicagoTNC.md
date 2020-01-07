## Textmining using the RCC Midway2 cluster

### Grabbing a node from midway2.rcc.uchicago.edu (after login)
Asking for 1 GPU, 1 CPU, 8GB RAM for 1 hour
```sh
sinteractive --nodes=1 --ntasks-per-node=1 --mem=8000 --time=1:00:00
```
This will usually grab a sandyb	(16 x Intel E5-2670 2.6GHz, 32 GB Ram). If necessary, use -p flag to grab another class of nodes.
See https://rcc.uchicago.edu/docs/using-midway/index.html for the types of nodes.

### Monitoring CPU, RAM usage
Once logged in, open a terminal to monitor the system performance.
```sh
xterm &
```
In the new terminal, run htop to monitor the CPU/RAM usage. Then press F4 for filter and enter the user id
```sh
htop
```

### Using the Anaconda3 python package
1. Load the necessary python modules -- using Anaconda allows us to install the required packages (e.g., unicodec, nltk). If you are not sure about the version, try 'module avail Anaconda3' 
```sh
module load Anaconda3/5.3.0
conda activate scipygeo18
```
1-2. If you have not setup your conda environment (i.e., scipygeo18), copy the geospatial_environment.yml file and run the below command. It sets up geopandas, geoplot, pysal, etc ...
```sh
conda env create -f geospatial_environment.yml
```

### Running jupyter notebook on the node
For jupyter notebook related instructions, see https://git.rcc.uchicago.edu/ivy2/Jupyter_on_compute_nodes

1. Load Anaconda3 module like above
2. Find out an ip address of the node:
```sh
(fastai) [kywch@midway2-gpu01 ~]$ ifconfig eno1 | grep 'inet '
        inet 10.50.221.191  netmask 255.255.252.0  broadcast 10.50.223.255
```
3. Start jupyter without launching a browser on the node (add & at the end to run in the background), using the ip address obtained above: 
```sh
(fastai) [kywch@midway2-gpu01 ~]$ jupyter notebook --no-browser --ip=10.50.221.191 &
```
4. By default, it would listen on port 8888. However, if the port is already taken by another user, it will complain. Try the next available port with --port=<port number> option.

5. Eventually, you'll get a URL with a token. For example: http://10.50.221.191:8888/?token=2af56958386caf78d5bdc2086b20b6ff18553b701c581cd3

6. Point the browser running on your laptop/desktop to this URL. Notice: compute nodes are only visible on internal uchicago network. Therefore, you either need to be on campus or use VPN to be able to use this method.

7. **To use GPU-accelerated libraries in jupyter notebook, use Python [conda env:DL_GPU] kernel.** (although, I'm not sure. The default kernel seems to use cuda. However, you can check -->) In the notebook, check whether it works by running.
```python
import torch
torch.cuda.is_available()       # Should be True
torch.backends.cudnn.enabled    # Should be True
```











