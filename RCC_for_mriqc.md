## Using mriqc on the RCC Midway cluster

### Grabbing a node from midway.rcc.uchicago.edu (after login)
Asking for 1 GPU, 8 CPU, 16GB RAM for 1 hour
```sh
sinteractive -p sandyb --nodes=1 --ntasks-per-node=8 --mem=16000 --time=1:00:00
```
This will grab a sandyb	(16 x Intel E5-2670 2.6GHz, 32 GB Ram). 

### Monitoring CPU, GPU, RAM usage
Once logged in, open a terminal to monitor the system performance.
```sh
xterm &
```
In the new terminal, run htop to monitor the CPU/RAM usage. Then press F4 for filter and enter the user id
```sh
htop
```

### Loading the necessary packages to use mriqc
```sh
module load python/3.6.1+intel-16.0
module unload mkl
module load afni
module load fsl
module load ANTs
```

### Running the mriqc commands (participant level)
You should replace RAWDATA_DIR, OUTPUT_DIR, and PARTICIPANT_ID with the actual values
For example, ~/NUBE_data/rawdata, ~/NUBE_data/mriqc, pilot01, 

```sh
$ mriqc RAWDATA_DIR OUTPUT_DIR participant --participant_label PARTICIPANT_ID --no-sub
```

### Make a symbolic link of the data directory to home directory
```sh
$ cd ~
$ ln -s /project2/bermanm/NUBE_data
```


