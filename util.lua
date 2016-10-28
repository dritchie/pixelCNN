
local function toDouble(imgData)
	return imgData:type() == 'torch.CudaLongTensor' and imgData:cudaDouble() or imgData:double()
end

local function toLong(imgData)
	return imgData:type() == 'torch.CudaTensor' and imgData:cudaLong() or imgData:long()
end

local function normalizeByteImage(imgData)
	return toDouble(imgData):div(256)
end

local function quantize(imgData, quantLevels)
	return toLong(imgData:mul(quantLevels):add(1))
end

local function dequantize(imgData, quantLevels)
	return toDouble(imgData):csub(1):div(quantLevels-1)
end

return {
	toDouble = toDouble,
	toLong = toLong,
	normalizeByteImage = normalizeByteImage,
	quantize = quantize,
	dequantize = dequantize
}