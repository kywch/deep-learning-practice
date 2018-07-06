## Textmining using the RCC Midway2 cluster

### Grabbing a node from midway2.rcc.uchicago.edu (after login)
Asking for 1 GPU, 1 CPU, 8GB RAM for 1 hour
```sh
sinteractive --nodes=1 --ntasks-per-node=1 --mem=8000 --time=1:00:00
```
This will usually grab a sandyb	(16 x Intel E5-2670 2.6GHz, 32 GB Ram). If necessary, use -p flag to grab another class of nodes.
See https://rcc.uchicago.edu/docs/using-midway/index.html for the types of nodes.

* used the bigmem2, 21 cores, 100 GB ram.

### Monitoring CPU, GPU, RAM usage
Once logged in, open a terminal to monitor the system performance.
```sh
xterm &
```
In the new terminal, run htop to monitor the CPU/RAM usage. Then press F4 for filter and enter the user id
```sh
htop
```

### Using the Anaconda3 python package
1. Load the necessary python modules -- using Anaconda allows us to install the required packages (e.g., unicodec, nltk)
```sh
module load Anaconda3/5.1.0
conda activate fastai
```

#### Installing the packages ####
1. nltk ( https://anaconda.org/anaconda/nltk ):
```sh
conda install -c anaconda nltk
```

2. unidecode ( https://anaconda.org/anaconda/unidecode ): 
```sh
conda install -c anaconda unidecode
```

3. pubmed parser ( https://github.com/titipata/pubmed_parser ): copy the pubmed_parser directory under the working directory

4. Network X ( https://anaconda.org/anaconda/networkx ) for network analyses using python
```sh
conda install -c anaconda networkx
```
