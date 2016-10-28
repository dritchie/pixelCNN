-- References:
--    https://github.com/kundan2510/pixelCNN
--    Figure 2 of https://arxiv.org/pdf/1606.05328v2.pdf
--    Table 1 of https://arxiv.org/pdf/1601.06759v3.pdf

-- A single layer of compute in a pixelCNN
local function pixelCNNLayer(nIn, nOut, filterSize, isFirstLayer)
	if isFirstLayer == nil then isFirstLayer = false end

	local vInput = nn.Identity()()
	local hInput = nn.Identity()()

	-- TODO: This padding/cropping scheme is slightly different than the one in the reference
	--    implementation (it uses two schemes: one for when vertical output is fed to horizontal
	--    stack, and one for when vertical output is returned). I'm not sure why they did that.
    --    AFAIK, the scheme below is correct, but I could be wrong...

    ------------------------ Vertical stack ------------------------

	-- Convolution padded such that first row of output depends on none of the input,
	--    second row depends only on first row, third row depends on first two, etc.
	local vConvOutput = nn.SpatialConvolution(nIn, 2*nOut,
											  filterSize, math.ceil(filterSize/2),
											  1, 1,
											  math.floor(filterSize/2), math.ceil(filterSize/2))(vInput)
	-- Have to crop the output to get rid of extraneous rows at the end
	vConvOutput = nn.SpatialZeroPadding(0, 0, 0, -(math.ceil(filterSize/2) + 1))(vConvOutput)

	-- Split the feature maps in half, to use in the gating function
	local vTanhInput = nn.Narrow(2, nOut+1, nOut)(vConvOutput)
	local vSigmoidInput = nn.Narrow(2, 1, nOut)(vConvOutput)

	-- Final vertical stack output is formed by the multiplication of the two gating functions
	local vFinalOutput = nn.CMulTable()({nn.Tanh()(vTanhInput), nn.Sigmoid()(vSigmoidInput)})


	------------------------ Horizontal stack ------------------------

	-- Another padded/cropped convolution, this time emulating an n x 1 filter
	local hConvOutput = nn.SpatialConvolution(nIn, 2*nOut,
											  math.ceil(filterSize/2), 1,
											  1, 1,
											  math.ceil(filterSize/2), 0)(hInput)
	-- How we crop depends on whether this is the first layer:
	--   Yes: crop as in the vertical case (column i depends on columns i-1, i-2, ...)
	--   No: also allow column i to depend on itself (result from previous layers)
	if isFirstLayer then
		hConvOutput = nn.SpatialZeroPadding(0, -(math.ceil(filterSize/2) + 1), 0, 0)(hConvOutput)
	else
		hConvOutput = nn.SpatialZeroPadding(-1, -math.ceil(filterSize/2), 0, 0)(hConvOutput)
	end

	-- Combine the horizontal 'masked' conv with the vertical masked conv (after putting
	--    the latter through an additional 1x1 conv)
	local combined = nn.CAddTable()({
		hConvOutput,
		nn.SpatialConvolution(2*nOut, 2*nOut, 1, 1)(vConvOutput)
	})

	-- Again, split feature maps, put them through gating functions
	local hTanhInput = nn.Narrow(2, nOut+1, nOut)(combined)
	local hSigmoidInput = nn.Narrow(2, 1, nOut)(combined)
	local hGatedOutput = nn.CMulTable()({nn.Tanh()(hTanhInput), nn.Sigmoid()(hSigmoidInput)})

	-- Final output uses residual connection to original h stack input
	-- UNLESS this is the first layer, b/c in that case the input has a different number
	--    of slices than the output and so they can't be added
	-- (Alternatively: could put the input through a 1x1 conv to give it the right number
	--    of slices...)
	local hFinalOutput
	if isFirstLayer then
		hFinalOutput = hGatedOutput
	else
		hFinalOutput = nn.CAddTable()({
			nn.SpatialConvolution(nOut, nOut, 1, 1)(hGatedOutput),
			hInput
		})
	end

	return nn.gModule({vInput, hInput}, {vFinalOutput, hFinalOutput})
end

-- For now, this only handles 1-channel (i.e. grayscale) images
--    (I think multi-channel requires modification to feed previous channels
--    into generation of subsequent channels)
-- Input tensor should be of shape (batchSize, 1, height, width)
local function pixelCNN(nFeatureMaps, nLayers, outNumClasses, initFilterSize, layerFilterSize)
	initFilterSize = initFilterSize or 7
	layerFilterSize = layerFilterSize or 3

	-- Initialize vertical and horizontal conv stacks
	local input = nn.Identity()()
	local firstLayer = pixelCNNLayer(1, nFeatureMaps, initFilterSize, true)
	local stacks = firstLayer({input, nn.Identity()(input)}) -- b/c nngraph complains if same input twice

	-- Do computation layers
	for i=1,nLayers do
		local layer = pixelCNNLayer(nFeatureMaps, nFeatureMaps, layerFilterSize, false)
		stacks = layer(stacks)
	end
	local output = nn.SelectTable(2)(stacks) -- top of the horizontal stack is final feature maps

	-- Finish up with a couple of ReLU + 1x1 convs
	-- (Last one squishes output to have outNumClasses slices)
	output = nn.ReLU()(output)
	output = nn.SpatialConvolution(nFeatureMaps, nFeatureMaps, 1, 1)(output)
	output = nn.ReLU()(output)
	output = nn.SpatialConvolution(nFeatureMaps, outNumClasses, 1, 1)(output)

	return nn.gModule({input}, {output})
end

return pixelCNN

