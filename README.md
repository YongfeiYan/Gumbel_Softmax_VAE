# Gumbel Softmax VAE

PyTorch implementation of a __Variational Autoencoder with Gumbel-Softmax Distribution__. 

*  Refer to the following paper: [Categorical Reparametrization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf) by Jang, Gu and Poole
* This implementation based on [dev4488's implementation](https://github.com/dev4488/VAE_gumble_softmax/blob/master/README.md) with the following modifications
  * Fixed KLD calculation
  * Fixed bug in calculating latent discrete probability
  * Fixed sampling distribution to get better images
  * Fixed training objective as the mean -ELBO in each batch which is consistent with [the author's implementation](https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb)
  * Refacted code for PyTorch 0.4.0 and cpu version



## Table of Contents
* [Installation](#installation)
* [Training](#train)
* [Results](#results)

## Installation

The program requires the following dependencies (easy to install using pip or Ananconda):

* python 3.6
* pytorch (version 0.4.0)
* numpy



### Training

```python
python gumbel_softmax_vae.py --log-interval 100 --epochs 100
```

## Results

Better training accuracy and sample image quality were obtained.

### Training output

```bash
Train Epoch: 1 [0/60000 (0%)]	Loss: 542.612183
Train Epoch: 1 [10000/60000 (17%)]	Loss: 209.495697
Train Epoch: 1 [20000/60000 (33%)]	Loss: 188.947800
Train Epoch: 1 [30000/60000 (50%)]	Loss: 195.400696
Train Epoch: 1 [40000/60000 (67%)]	Loss: 189.618195
Train Epoch: 1 [50000/60000 (83%)]	Loss: 190.018005
====> Epoch: 1 Average loss: 199.9627
====> Test set loss: 182.4984
Train Epoch: 2 [0/60000 (0%)]	Loss: 181.569107
Train Epoch: 2 [10000/60000 (17%)]	Loss: 174.215042
Train Epoch: 2 [20000/60000 (33%)]	Loss: 169.722961
Train Epoch: 2 [30000/60000 (50%)]	Loss: 169.356277
Train Epoch: 2 [40000/60000 (67%)]	Loss: 161.219177
Train Epoch: 2 [50000/60000 (83%)]	Loss: 155.769821
====> Epoch: 2 Average loss: 164.6240
```



### MNIST
| Training Step |  Ground Truth/Reconstructions   |    Generated Samples    |
| ------------- | :-----------------------------: | :---------------------: |
| 1             | ![](data/reconstruction_1.png)  | ![](data/sample_1.png)  |
| 10            | ![](data/reconstruction_10.png) | ![](data/sample_10.png) |
| 20            | ![](data/reconstruction_20.png) | ![](data/sample_20.png) |
| 30            | ![](data/reconstruction_30.png) | ![](data/sample_30.png) |