## Running matlab using the RCC Midway2 cluster

### Grabbing a node from midway2.rcc.uchicago.edu (after login)
Asking for a node exclusively for an hour
```sh
sinteractive -p broadwl --nodes=1 --time=1:00:00 --exclusive
```
If necessary, use -p flag to grab another class of nodes.
See https://rcc.uchicago.edu/docs/using-midway/index.html for the types of nodes.


### Monitoring CPU, GPU, RAM usage
Once logged in, open a terminal to monitor the system performance.
```sh
xterm &
```
In the new terminal, run htop to monitor the CPU/RAM usage. Then press F4 for filter and enter the user id
```sh
htop
```

### Loading the matlab module ###
1. Check the available matlab versions
```sh
module avail matlab
```

2. Load the matlab (the below command will load the default version, as of 2019/04, it's matlab 2017b)
```sh
module load matlab
```

3. Run the matlab (the below command will launch the matlab GUI)
```sh
matlab
```

