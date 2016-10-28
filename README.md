# pixelCNN
Torch implementation of PixelCNN for MNIST.

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
