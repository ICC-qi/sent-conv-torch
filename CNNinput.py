
str1 = raw_input("Please input:")
f=open('/home/icc-qi/sent-conv-torch-master/MR_word_mapping.txt','r')
fout=open('/home/icc-qi/sent-conv-torch-master/CNNinput.txt','w')
voc={}
for line in f.readlines():                          
    line = line.strip().split(' ')
    word = line[0]
    index = line[1]
    voc[word] = index   
str1=str1.lower().split(' ')
output=[1]*64
fout.write('torch.Tensor{')
for i in range(len(str1)):
    output[i+4]=voc[str1[i]]
for j in range(64):
    fout.write(str(output[j])+',')
fout.write('}')
print output
f.close()
fout.close()
print 'done'
