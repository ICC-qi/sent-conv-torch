require 'cunn';
require 'cutorch';
require 'nn';
require 'nngraph';
require 'cudnn';
require 'hdf5';

-- Read model
--model=torch.load('/home/icc-qi/sent-conv-torch-master/results1/20180514_1644_model_1.t7').model
model=torch.load('/home/icc-qi/sent-conv-torch-master/resultsH04/20180615_1534_model_4.t7').model

-- Read HDF5 training data
fr = hdf5.open('/home/icc-qi/sent-conv-torch-master/H04.hdf5', 'r')
train = fr:read('train'):all()
train_label = fr:read('train_label'):all()
print(train:size())

--hdf5 write files
myFile = hdf5.open('/home/icc-qi/sent-conv-torch-master/H04featurevector.hdf5', 'w')

--input=torch.Tensor{1,1,1,1,74,352,131,82,269,98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}:reshape(1,64):cuda()
--output=model:forward(input)
--print(output)

-- Model process
model1=model:clone()
model1.modules[21]=nil
model1.modules[20]=nil
model1.modules[19]=nil
--torch.save('model1.t7',model1)
print(model1:size())

--Get vector
for i=1,train:size()[1] do
input=model1.modules[2]:forward(train[i]:reshape(1,train:size()[2]):cuda())
--3
a=model1.modules[3]:forward(input)
a=model1.modules[4]:forward(a)
a=model1.modules[5]:forward(a)
a=model1.modules[6]:forward(a)
a=model1.modules[7]:forward(a)
--4
b=model1.modules[8]:forward(input)
b=model1.modules[9]:forward(b)
b=model1.modules[10]:forward(b)
b=model1.modules[11]:forward(b)
b=model1.modules[12]:forward(b)
--5
c=model1.modules[13]:forward(input)
c=model1.modules[14]:forward(c)
c=model1.modules[15]:forward(c)
c=model1.modules[16]:forward(c)
c=model1.modules[17]:forward(c)
output=model1.modules[18]:forward{a,b,c}
--print(output:size())
--print(train[i])
--print(output)
--assert(1==0)
if i%500==0 then
print('No.' ..i)
print(output:size())
end
label='H04C' ..train_label[i] ..'I' ..i
myFile:write(label, output:double())
--assert(1==0)
end

fr:close()
myFile:close()
