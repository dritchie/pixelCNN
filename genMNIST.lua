
require('./globals')
local util = require('./util')

-- How to quantize the images
-- 2 = binary images
local numQuantBins = 2

local model = torch.load(paths.concat(paths.cwd(), 'mnistPixelCNN.net'))

local gpu = true
local softmax = nn.SoftMax()
if gpu then
    cudnn.convert(model, cudnn)
    model = model:cuda()
    cudnn.convert(softmax)
    softmax = softmax:cuda()
    print('')
end
local TensorType = gpu and torch.CudaTensor or torch.Tensor
local QuantTensorType = gpu and torch.CudaLongTensor or torch.LongTensor

-- Multinomial sampling for 3D tensors
-- result should be size (batchSize x height x width)
-- prob should be size (batchSize x numClasses x height x width)
local function multinomial(prob)
	-- Re-order so actual probs are last, flatten into a flat list of probs
	local flatProbs = prob:permute(1, 3, 4, 2):contiguous()
	flatProbs = flatProbs:view(flatProbs:size(1)*flatProbs:size(2)*flatProbs:size(3), flatProbs:size(4))
	-- Sample once from multinomial
	local samp = torch.multinomial(flatProbs, 1, true)
	-- Unflatten before returning
	return samp:view(prob:size(1), prob:size(3), prob:size(4))
end

local function generate(n)
	local x = TensorType(n, 1, 28, 28)
	for i=1,28 do		-- rows
		for j=1,28 do	-- cols
			-- Get (batchSize x numClasses x height x width) network outputs
			local out = model:forward(x)
			-- Turn 'em into probabilities
			out = softmax:forward(out)
			-- Draw a multinomial sample per pixel
			local samp = multinomial(out)
			-- Convert quantized samples back to floating-point (-1 to deal with 1-based indexing)
			local fpsamp = util.dequantize(samp, numQuantBins)
			-- Write the current pixel from the dequantized sample back into x
			x[{{}, 1, i, j}] = fpsamp[{{}, i, j}]
		end
	end
	local outx = torch.Tensor(n, 1, 28, 28):copy(x)
	return outx
end

local n = 10
local samps = generate(10)
for i=1,n do
	local filename = paths.concat(paths.cwd(), string.format('sample_%03d.png', i))
	image.save(filename, samps[i])
end