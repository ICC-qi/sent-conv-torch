import jieba
import jieba.analyse
f=open('/home/icc-qi/sent-conv-torch-master/torch/data_set_abs_section','r')
f1=open('/home/icc-qi/sent-conv-torch-master/torch/data_set_abs_section_jieba15','w')
lines=f.readlines()
for line in lines:
    class1=line.split(' ')[0]
    f1.write(class1+' ')
    content=line[len(class1):].strip()
    #jieba.analyse.set_stop_words("../extra_dict/stop_words.txt")
    #jieba.analyse.set_idf_path("../extra_dict/idf.txt.big");
    tags=jieba.analyse.extract_tags(content, topK=15, withWeight=True)
    for tag in tags:
        #print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
        f1.write(tag[0]+' ')
    f1.write('\n')
f.close()
f1.close()    
