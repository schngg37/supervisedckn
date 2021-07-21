
## Running on Google Colab 
(following documentation is assuming that 
all our codes and files are in folder 'ckn-sachin' of google drive)

Start by mounting drive 
```
from google.colab import drive
drive.mount('/content/gdrive')
```

Now install the packages using the following commands

```
import numpy
import scipy
import sklearn
import torch
print(torch.__version__)
```
Now change the working directory as follows
```
%cd ~
%cd /content/gdrive/My Drive/ckn-sachin
%cd third-party
!ls
%cd miso_svm-1.0
!ls
```
The Python package `miso_svm` can be installed with
 (original [repository][https://gitlab.inria.fr/gdurif/ckn-tf/tree/prod/miso_svm/])
```
%cd miso_svm-1.0
!ls
!python "setup.py" install
```
Change the directory
```
%cd ~
%cd /content/gdrive/My Drive/ckn-sachin
```

## Results

The results from the original paper (Mairal, 2016) were achieved using
cudnn-based Matlab code available [here][https://gitlab.inria.fr/mairal/ckn-cudnn-matlab/]. 

To run the following experiments, please first download the 
[data][http://pascal.inrialpes.fr/data2/mairal/data/cifar_white.mat], 
put into the folder `./data/cifar-10` and then do.


#### Unsupervised CKN

Here is a summary of the results of **unsupervised** CKN on CIFAR10 image classification dataset
with pre-whitening and without data augmentation or model ensembling.

```bash
# Code examples
python cifar10_unsup.py --filters 64 256 --subsamplings 2 6 --kernel-sizes 3 3
```

#### Supervised CKN

Here is a summary of the results of **supervised** CKN on CIFAR10 image classification dataset 
with pre-whitening and without data augmentation or model ensembling.

```bash
# Code examples
python cifar10_sup.py --epochs 105 --lr 0.1 --alpha 0.001 --loss hinge --alternating --model ckn5
python cifar10_sup.py --epochs 105 --lr 0.1 --alpha 0.1 --loss hinge --alternating --model ckn14
```
To obtain the training curves, code is added at last of 'cifar10_sup.py' which save thes figures in './ckn-sachin/output'
