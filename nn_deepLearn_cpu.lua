-- deps
require 'nn';
require 'paths';

-- grab the data
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile','bird','cat','deer','dog',
            'frog','horse','ship','truck'}

-- clean the training data
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

mean = {}
stdv  = {}
for i=1,3 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean()
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

-- build the nn
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))
net:add(nn.Linear(16*5*5,120))
net:add(nn.ReLU())
net:add(nn.Linear(120,84))
net:add(nn.ReLU())
net:add(nn.Linear(84,10))
net:add(nn.LogSoftMax())

-- define the loss function
criterion = nn.ClassNLLCriterion()

-- train the net
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5

timer = torch.Timer()
trainer:train(trainset)
print(timer:time().real)

-- clean test data
testset.data = testset.data:double()
for i=1,3 do
    testset.data[{ {},{i}, {}, {}}]:add(-mean[i])
    testset.data[{ {}, {i}, {}, {}}]:div(stdv[i])
end

-- predictions and accuracy
correct = 0
for i = 1, 10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indicies = torch.sort(prediction, true)
    if groundtruth == indicies[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

-- which classes performed well?
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i = 1, 10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i = 1, #classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end









