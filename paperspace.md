### Running a jupyter server on a paperspace box
ssh into the server and run
```sh
conda activate fastai
jupyter notebook
```

https://184.105.190.246:7890 

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







