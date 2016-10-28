# pixelCNN
Torch implementation of PixelCNN for MNIST. Implements the network described in [these](https://arxiv.org/abs/1601.06759) [papers](https://arxiv.org/abs/1606.05328). kundan2510's [Theano implementation](https://github.com/kundan2510/pixelCNN) was an invaluable reference.

Requires `cutorch` and `cudnn`.

Also requires the MNIST dataset (duh!):
```
luarocks install mnist
```
Train the model (this will automatically save the model every so often):
```
th trainMNIST.lua
```
Generate images using the saved model:
```
th genMNIST.lua
```
