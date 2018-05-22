--do not use nngraph
require 'cudnn';
require 'hdf5';
require 'nn';
require 'cunn';
require 'cutorch';
require 'nngraph';
input=torch.Tensor{1,1,1,1,74,352,131,82,269,98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}:reshape(1,64):cuda()

model=nn.Sequential()
chan = nn.LookupTable(18766, 300)
model:add(chan)

a=nn.Sequential()
b=nn.Sequential()
c=nn.Sequential()

a:add(nn.Reshape(1, 64, 300, true))
a:add(nn.SpatialConvolution(1, 100, 300, 3))
a:add(nn.Reshape(100, 62, true))
a:add(nn.ReLU())
a:add(nn.Max(3))

b:add(nn.Reshape(1, 64, 300, true))
b:add(nn.SpatialConvolution(1, 100, 300, 4))
b:add(nn.Reshape(100, 61, true))
b:add(nn.ReLU())
b:add(nn.Max(3))

c:add(nn.Reshape(1, 64, 300, true))
c:add(nn.SpatialConvolution(1, 100, 300, 5))
c:add(nn.Reshape(100, 60, true))
c:add(nn.ReLU())
c:add(nn.Max(3))

--conv=nn.ParallelTable()
conv=nn.ConcatTable()
conv:add(a)
conv:add(b)
conv:add(c)

model:add(conv)
model:add(nn.JoinTable(2))
model=model:cuda()

model_old=torch.load('/home/icc-qi/sent-conv-torch-master/results1/20180514_1644_model_1.t7').model
para=model_old:parameters()
w2v=model_old.modules[2].weight
model:get(1).weight=w2v
model:get(2):get(1):get(2).weight=para[2]
model:get(2):get(1):get(2).bias=para[3]
model:get(2):get(2):get(2).weight=para[4]
model:get(2):get(2):get(2).bias=para[5]
model:get(2):get(3):get(2).weight=para[6]
model:get(2):get(3):get(2).bias=para[7]

torch.save('rebuild.t7',model)
print(model)
print(model:forward(input))
