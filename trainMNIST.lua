-- Reference: http://rnduja.github.io/2015/10/13/torch-mnist/

require('./globals')
local util = require('./util')
local mnist = require('mnist')
local pixelCNN = require('./pixelCNN')

local fullset = mnist.traindataset()

-- How to quantize the images
-- 2 = binary images
local numQuantBins = 2

local trainset = {
    size = 50000,
    data = util.normalizeByteImage(fullset.data[{{1,50000}}])
}
-- Quantized version of the data to be used as training "class labels"
trainset.quantData = util.quantize(trainset.data:clone(), numQuantBins)
-- Make training data be the dequantized version of this (for consistency at prediction time)
trainset.data = util.dequantize(trainset.quantData, numQuantBins)

-- Various parameters
local batchSize = 16
local checkpointEvery = 500     -- How often to save model to disk
local nFeatureMaps = 32
local nLayers = 6

-- Load model, convert to CuDNN version
local model = pixelCNN(nFeatureMaps, nLayers, numQuantBins)
cudnn.convert(model, cudnn)
model = model:cuda()
print('')

-- Loss
local criterion = cudnn.SpatialCrossEntropyCriterion():cuda()

-- Options for optimizer
local optimState = {
    learningRate = 0.001
}
local optMethod = optim.adam

local params, gradParams = model:getParameters()

-- Run one training batch
local function batch(indexPerm, t)
    -- setup inputs and targets for this mini-batch
    local size = math.min(t + batchSize, trainset.size) - t
    local inputs = torch.CudaTensor(size, 1, 28, 28)
    local quantInputs = torch.CudaLongTensor(size, 28, 28)
    for i = 1,size do
        local input = trainset.data[indexPerm[i+t]]
        inputs[i][1]:copy(input)
        local qinput = trainset.quantData[indexPerm[i+t]]
        quantInputs[i]:copy(qinput)
    end

    local function feval(x)
        model:zeroGradParameters()
        local loss = criterion:forward(model:forward(inputs), quantInputs)
        model:backward(inputs, criterion:backward(model.output, quantInputs))
        return loss, gradParams
    end

    local _, fs = optMethod(feval, params, optimState)

    return fs[1]
end

-- Run one training epoch
local function epoch()
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    
    for t = 1,trainset.size,batchSize do
        local loss = batch(shuffle, t)
        count = count + 1
        current_loss = current_loss + loss
        if count % checkpointEvery == 0 then
            torch.save(paths.concat(paths.cwd(), 'mnistPixelCNN.net'), model)
            print(string.format('Batch: %d | loss: %4f (checkpoint)', count, loss))
        else
            print(string.format('Batch: %d | loss: %4f', count, loss))
        end
    end

    -- normalize loss
    return current_loss / count
end

local function train(iters)
    for i = 1,iters do
        local loss = epoch()
        print(string.format('Epoch: %d | Current loss: %4f', i, loss))
    end
end

train(60)

