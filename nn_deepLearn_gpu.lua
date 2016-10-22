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
net:add(nn.SpatialConvolution(3,6,5,5)) -- change these params to make the net heavier
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(6,16,5,5)) -- change these params to make the net heavier
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

-- migrate tensors to cuda
require 'cunn';
net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

-- train the net
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5

timer = torch.Timer()
trainer:train(trainset)
print(timer:time().real)









