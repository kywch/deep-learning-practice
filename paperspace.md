### Creating a paperspace box

If you create a paperspace account through [this link](https://www.paperspace.com/&R=UBZSZHP), you will receive $10 in credit immediately after adding a valid payment method!!

---

### Running a jupyter server on a paperspace box
ssh into the server and run
```sh
conda activate fastai
jupyter notebook
```

https://184.105.5.189:7890/ 

Testing whether the jupyter notebook has access to cuda
```python
import torch
torch.cuda.is_available()       # Should be True
torch.backends.cudnn.enabled    # Should be True
```

---

### Installing fastai: https://github.com/fastai/fastai

First, delete the existing fastai directory and do a clean install by doing the following:

Recommended installation approach is to clone fastai using `git`:

```sh
git clone https://github.com/fastai/fastai.git
```
Then, `cd` to the fastai folder and create the python environment:

```sh
cd fastai
conda env update
```
This downloads all of the dependencies and then all you have to do is:

```sh
conda activate fastai
```

To update everything at any time, cd to your repo and:

```sh
git pull
conda env update
```

Some troubleshooting: follow this: https://medium.com/@GuruAtWork/fast-ai-lesson-1-7fc38e978d37

#### Advanced jupyter setup (e.g., using jupyter on https without the token)
Generate the config file and follow http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#running-a-public-notebook-server

For paperspace (https://www.paperspace.com), we have to set UFW to allow us accessing the port of the notebook server. This can be done with the following command. [Reference](https://by-the-water.github.io/posts/2017/05/16/setting-up-a-jupyter-notebook-server-on-paperspace.html) 
```sh
sudo ufw allow [your notebook server port]
```
Then we can access jupyter notebook from anywhere using a simplified link: https://[your public IP]:[your port] 

#### Downloading the Places2 dataset without storing it (takes 3~4 hours)
wget -qO- http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar | tar xv > /dev/null

---
### paperspace SSD storage vs. midway storage
```sh
paperspace@ps8hdvfpm:~/places365_standard$ time tree | wc -l
1840697

real	0m11.775s
user	0m7.504s
sys	0m3.356s
```

```sh
(fastai) [kywch@midway2-gpu02 places365_standard]$ time tree | wc -l
1840697

real    9m34.287s
user    0m5.272s
sys     0m39.172s
```


---
### Performance benchmarks: Places365 training (1.8M images)
Using a gpu2 node: 1 Tesla K80 gpu and 8 CPU cores
The gpu runs at 99% level

* Resnet18 + fastai + pytorch, single gpu (https://github.com/kywch/places365/blob/master/create_sbatch.ipynb): takes 0.83 s for a mini-batch (size 128) vs. 0.17 s on Paperspace Volta (V100)

*RCC Midway, K80*
```sh
Epoch: [0][100/14090]   Time 0.819 (0.928)      Data 0.000 (0.055)      Loss 5.6399 (6.0588)    Prec@1 0.781 (0.897)    Prec@5 7.812 (3.759)
Epoch: [0][200/14090]   Time 0.828 (0.877)      Data 0.000 (0.028)      Loss 5.4151 (5.8309)    Prec@1 0.781 (1.279)    Prec@5 8.594 (5.259)
Epoch: [0][300/14090]   Time 0.826 (0.861)      Data 0.000 (0.019)      Loss 5.3681 (5.6859)    Prec@1 2.344 (1.695)    Prec@5 3.906 (6.582)
Epoch: [0][400/14090]   Time 0.828 (0.852)      Data 0.000 (0.014)      Loss 5.0056 (5.5697)    Prec@1 6.250 (2.094)    Prec@5 14.062 (7.828)
Epoch: [0][500/14090]   Time 0.822 (0.847)      Data 0.000 (0.011)      Loss 5.0762 (5.4800)    Prec@1 3.125 (2.444)    Prec@5 10.156 (8.960)
Epoch: [0][600/14090]   Time 0.831 (0.844)      Data 0.001 (0.010)      Loss 4.9700 (5.4014)    Prec@1 3.906 (2.771)    Prec@5 13.281 (9.991)
Epoch: [0][700/14090]   Time 0.835 (0.842)      Data 0.000 (0.008)      Loss 4.8096 (5.3281)    Prec@1 7.812 (3.102)    Prec@5 17.969 (11.014)
Epoch: [0][800/14090]   Time 0.830 (0.840)      Data 0.000 (0.007)      Loss 4.5088 (5.2656)    Prec@1 10.156 (3.435)   Prec@5 25.781 (11.999)
Epoch: [0][900/14090]   Time 0.823 (0.839)      Data 0.000 (0.007)      Loss 4.7752 (5.2078)    Prec@1 7.031 (3.755)    Prec@5 18.750 (12.878)
Epoch: [0][1000/14090]  Time 0.828 (0.838)      Data 0.000 (0.006)      Loss 4.5860 (5.1541)    Prec@1 9.375 (4.119)    Prec@5 25.000 (13.813)
Epoch: [0][1100/14090]  Time 0.831 (0.837)      Data 0.000 (0.005)      Loss 4.5316 (5.1061)    Prec@1 5.469 (4.439)    Prec@5 21.094 (14.719)
Epoch: [0][1200/14090]  Time 0.830 (0.836)      Data 0.000 (0.005)      Loss 4.4343 (5.0614)    Prec@1 11.719 (4.756)   Prec@5 32.031 (15.520)
Epoch: [0][1300/14090]  Time 0.832 (0.836)      Data 0.000 (0.005)      Loss 4.4053 (5.0212)    Prec@1 7.812 (5.066)    Prec@5 24.219 (16.262)
Epoch: [0][1400/14090]  Time 0.822 (0.835)      Data 0.000 (0.004)      Loss 4.5136 (4.9809)    Prec@1 10.938 (5.361)   Prec@5 26.562 (17.019)
Epoch: [0][1500/14090]  Time 0.824 (0.835)      Data 0.000 (0.004)      Loss 4.3716 (4.9420)    Prec@1 11.719 (5.668)   Prec@5 28.125 (17.733)
Epoch: [0][1600/14090]  Time 0.830 (0.834)      Data 0.000 (0.004)      Loss 4.3123 (4.9083)    Prec@1 7.812 (5.906)    Prec@5 29.688 (18.389)
Epoch: [0][1700/14090]  Time 0.825 (0.834)      Data 0.000 (0.004)      Loss 4.4168 (4.8716)    Prec@1 9.375 (6.198)    Prec@5 26.562 (19.091)
Epoch: [0][1800/14090]  Time 0.837 (0.834)      Data 0.000 (0.004)      Loss 4.3819 (4.8382)    Prec@1 10.938 (6.459)   Prec@5 25.781 (19.733)
Epoch: [0][1900/14090]  Time 0.841 (0.834)      Data 0.000 (0.003)      Loss 4.0898 (4.8086)    Prec@1 12.500 (6.711)   Prec@5 34.375 (20.328)
Epoch: [0][2000/14090]  Time 0.820 (0.833)      Data 0.000 (0.003)      Loss 3.8935 (4.7786)    Prec@1 17.188 (6.997)   Prec@5 41.406 (20.930)
Epoch: [0][2100/14090]  Time 0.838 (0.833)      Data 0.000 (0.003)      Loss 4.0312 (4.7496)    Prec@1 13.281 (7.240)   Prec@5 33.594 (21.508)
```

*Paperspace, Volta 100*
* Using single precision can double the speed (https://www.xcelerit.com/computing-benchmarks/insights/benchmarks-deep-learning-nvidia-p100-vs-v100-gpu), BUT then V100 is so fast, so the bottleneck is the data fetching. So for our training, 8 CPUs are working 90+% and the GPU is going back and forth between 0 and 100%. 
```sh
Epoch: [0][10/14090]    Time 0.142 (0.658)      Data 0.001 (0.201)      Loss 6.5146 (6.8311)    Prec@1 0.000 (0.284)      Prec@5 0.000 (0.994)
Epoch: [0][20/14090]    Time 0.142 (0.412)      Data 0.001 (0.106)      Loss 6.1472 (6.6016)    Prec@1 0.000 (0.298)      Prec@5 3.906 (1.711)
Epoch: [0][30/14090]    Time 0.146 (0.325)      Data 0.001 (0.072)      Loss 6.1444 (6.4577)    Prec@1 1.562 (0.378)      Prec@5 4.688 (1.890)
Epoch: [0][40/14090]    Time 0.147 (0.281)      Data 0.001 (0.054)      Loss 6.1019 (6.3633)    Prec@1 0.000 (0.476)      Prec@5 3.125 (2.001)
Epoch: [0][50/14090]    Time 0.139 (0.253)      Data 0.001 (0.045)      Loss 5.9633 (6.2962)    Prec@1 0.000 (0.444)      Prec@5 1.562 (2.068)
Epoch: [0][60/14090]    Time 0.138 (0.238)      Data 0.084 (0.048)      Loss 5.8989 (6.2316)    Prec@1 1.562 (0.487)      Prec@5 3.125 (2.331)
Epoch: [0][70/14090]    Time 0.143 (0.228)      Data 0.001 (0.053)      Loss 5.8209 (6.1810)    Prec@1 0.781 (0.583)      Prec@5 3.125 (2.487)
Epoch: [0][80/14090]    Time 0.137 (0.217)      Data 0.046 (0.052)      Loss 5.7792 (6.1311)    Prec@1 2.344 (0.781)      Prec@5 2.344 (2.865)
Epoch: [0][90/14090]    Time 0.320 (0.214)      Data 0.258 (0.057)      Loss 5.6345 (6.0933)    Prec@1 2.344 (0.850)      Prec@5 6.250 (3.091)
Epoch: [0][100/14090]   Time 0.141 (0.208)      Data 0.000 (0.056)      Loss 5.6495 (6.0600)    Prec@1 1.562 (0.866)      Prec@5 6.250 (3.210)
Epoch: [0][110/14090]   Time 0.140 (0.203)      Data 0.030 (0.054)      Loss 5.7711 (6.0286)    Prec@1 0.781 (0.950)      Prec@5 7.812 (3.519)
Epoch: [0][120/14090]   Time 0.141 (0.199)      Data 0.000 (0.052)      Loss 5.5604 (6.0007)    Prec@1 0.781 (0.981)      Prec@5 7.031 (3.796)
Epoch: [0][130/14090]   Time 0.138 (0.196)      Data 0.001 (0.050)      Loss 5.5680 (5.9755)    Prec@1 2.344 (1.026)      Prec@5 10.156 (3.984)
Epoch: [0][140/14090]   Time 0.139 (0.193)      Data 0.044 (0.050)      Loss 5.7192 (5.9511)    Prec@1 0.000 (1.053)      Prec@5 7.812 (4.167)
Epoch: [0][150/14090]   Time 0.143 (0.191)      Data 0.001 (0.050)      Loss 5.6044 (5.9285)    Prec@1 0.781 (1.076)      Prec@5 7.031 (4.377)
Epoch: [0][160/14090]   Time 0.455 (0.190)      Data 0.394 (0.050)      Loss 5.5062 (5.9084)    Prec@1 0.000 (1.097)      Prec@5 5.469 (4.508)
Epoch: [0][170/14090]   Time 0.145 (0.189)      Data 0.001 (0.051)      Loss 5.5245 (5.8864)    Prec@1 0.000 (1.124)      Prec@5 9.375 (4.665)
Epoch: [0][180/14090]   Time 0.136 (0.187)      Data 0.025 (0.050)      Loss 5.4950 (5.8662)    Prec@1 3.125 (1.148)      Prec@5 8.594 (4.834)
Epoch: [0][190/14090]   Time 0.138 (0.186)      Data 0.001 (0.050)      Loss 5.4719 (5.8479)    Prec@1 0.781 (1.162)      Prec@5 3.906 (4.941)
Epoch: [0][200/14090]   Time 0.142 (0.185)      Data 0.032 (0.050)      Loss 5.3288 (5.8316)    Prec@1 4.688 (1.178)      Prec@5 11.719 (5.041)
```





