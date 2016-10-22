batchSize, inputSize, outputSize = 15000, 15000, 15000
input = torch.FloatTensor(batchSize, inputSize):uniform(0,1)
weight = torch.FloatTensor(outputSize, inputSize):uniform(0,1)
output = torch.FloatTensor(batchSize, outputSize)

timer = torch.Timer()
output:addmm(0, output, 1, input, weight:t())
print(timer:time().real)

require 'cutorch';
input = torch.CudaTensor(batchSize, inputSize):uniform(0,1)
weight = torch.CudaTensor(outputSize, inputSize):uniform(0,1)
output = torch.CudaTensor(batchSize, outputSize)

time = timer:reset()
output:addmm(0, output, 1, input, weight:t())
print(timer:time().real)

