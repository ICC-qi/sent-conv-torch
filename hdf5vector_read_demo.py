import h5py
myFile = h5py.File('/home/icc-qi/sent-conv-torch-master/LDAH04featurevector_dropout0.hdf5', 'r')
classFile = h5py.File('/home/icc-qi/sent-conv-torch-master/LDAH04_8topic.hdf5', 'r')


def distance(a,b):
  summary=0
  if len(a)==len(b):
    for i in range(len(a)):
      summary+=pow(a[i]-b[i],2)
    return pow(summary,0.5)
  else:
    assert(1==0)

# The '...' means retrieve the whole tensor
#myFile.keys()
#data = myFile['H04C1I1'][0]
a=[0,1,2]
b=[1,3,4]
print('test: '+str(distance(a,b)))

class1=classFile['train_label']
num_file=len(class1)  #94034
#sum1 = [[0] for i in range(8)]
sum1=[0,0,0,0,0,0,0,0]
num1=[0,0,0,0,0,0,0,0]
#data1 = myFile['I1'][0]
#avg=[]

#the same text
m=10
n='I'+str(m+1)
data3 = myFile[n][0]
print('data'+str(m)+' class: '+str(class1[m]-1))
#data11 = myFile['I11'][0]
#data12 = myFile['I12'][0]
#data867 = myFile['I867'][0]
#data6979 = myFile['I6979'][0]
#data61944 = myFile['I61944'][0]
#print(data)
#dif=distance(data3,data1)
#sim=distance(data3,data867)
#print('data3 and data1: '+str(dif))
#print('data3 and data867: '+str(sim))
#print('data3 and data6979: '+str(distance(data3,data6979)))
#print('data3 and data61944: '+str(distance(data3,data61944)))
#print('data3 and data11: '+str(distance(data3,data11)))
#print('data3 and data12: '+str(distance(data3,data12)))

for i in range(num_file):
  label='I'+str(i+1)
  vec=myFile[label][0]
  dif=distance(data3,vec)
  sum1[class1[i]-1]+=dif
  num1[class1[i]-1]+=1

for j in range(8):
  if not num1[j]==0:
    avg=sum1[j]/num1[j]
    print('class '+str(j)+' avg distance: '+str(avg))

myFile.close()
classFile.close()
