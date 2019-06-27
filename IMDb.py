# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import urllib.request
import os
import tarfile

'''
下载解压数据
'''
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="C:/Users/Administrator/pythonwork/aclImdb_vl.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
    
if not os.path.exists("C:/Users/Administrator/pythonwork/aclImdb"):
    tfile=tarfile.open(filepath,'r:gz')
    result=tfile.extractall('C:/Users/Administrator/pythonwork/')
    
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

'''
创建rm_tags函数删除文字中的html标签
'''
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

'''
创建read_files函数读取IMDb文件目录，前12500个正面，后12500个负面
'''
def read_files(filetype):
    path = "C:/Users/Administrator/pythonwork/aclImdb/"
    file_list=[]
    
    positive_path=path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
    
    all_labels = ([1]*12500 + [0]*12500)
    
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_labels,all_texts

y_train,train_text=read_files("train")
y_test,test_text=read_files("test")



'''
建立token
'''
token = Tokenizer(num_words=2000)                       #建立一个有2000个单词的字典
token.fit_on_texts(train_text)                          #读取所有训练数据，按单词出现次数排序，前2000名的单词列入字典

print(token.document_count)                             #token读取了多少文章
print(token.word_index)

x_train_seq = token.texts_to_sequences(train_text)      #影评文字转换为数字列表  token.texts_to_sequences
x_test_seq = token.texts_to_sequences(test_text) 

x_train = sequence.pad_sequences(x_train_seq, maxlen=100) #截长补短使数字列表长度等于100
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)


'''
建立多层感知器模型进行IMDB情感分析
'''
from keras.datasets import imdb

'''
加入嵌入层
'''
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
'''
建立模型
'''
model = Sequential()                           
'''
嵌入层
'''
model.add(Embedding(output_dim=32,                           #将数字列表转换为32维向量
                    input_dim=2000,                          #即单词字典长度
                    input_length=100))                       #即maxlen
model.add(Dropout(0.2))                                      #每次训练迭代随机放弃20%神经元，以避免过拟合

'''
建立多层感知器模型
''' 
'''
平坦层
'''
model.add(Flatten())                                   #平坦层共3200个神经元，数字列表每一个数字都转换为32维向量
'''
隐藏层
'''
model.add(Dense(units=256,                             #隐藏层共256个神经元      
                activation='relu'))                    #定义激活函数ReLU
model.add(Dropout(0.35))
'''
输出层
'''
model.add(Dense(units=1,                               #隐藏层共1个神经元，输出1正面评价，输出0负面评价     
                activation='sigmoid'))                 #定义激活函数Sigmoid

model.summary()                                        #查看模型摘要

'''
定义训练模型
'''
model.compile(loss='binary_crossentropy',           
              optimizer='adam',
              metrics=['accuracy'])
'''
开始训练
'''
train_history = model.fit(x_train,y_train,batch_size=100,    #每一批次100项数据
                          epochs=10,verbose=2,               #执行10个训练周期，verbose=2显示训练过程
                          validation_split=0.2)              #训练与验证数据比例，80%作为训练数据，20%验证

'''
评估模型准确率
'''
scores = model.evaluate(x_test, y_test, verbose=1)         #评估模型准确率,参数为特征值、标签值
scores[1]                                                  #显示准确率
 
'''
进行预测
'''
predict=model.predict_classes(x_test)                      # model.predict_classes进行预测，参数features
predict_classes=predict.reshape(-1)                        # predict是二维数组，转换为一维

'''
创建display_test_Sentiment函数，查看测试数据预测结果
'''
SentimentDict={1:"正面的",0:"负面的"}
def display_test_Sentiment(i):
    print(test_text[i])
    print('label真实值',SentimentDict[y_test[i]],
          '预测结果',SentimentDict[predict_classes[i]])

'''
创建predict_review()函数，传入input_text（影评文字）就可以预测为正面或负面
'''
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen=100)
    predict_result=model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])
    

'''
RNN模型进行预测
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model2 = Sequential()                           
model2.add(Embedding(output_dim=32,                           #将数字列表转换为32维向量
                    input_dim=2000,                           #即单词字典长度
                    input_length=100))                        #即maxlen
model2.add(Dropout(0.2))                                      #每次训练迭代随机放弃20%神经元，以避免过拟合

model2.add(SimpleRNN(units=16))                                #RNN层

model2.add(Dense(units=256,                                    #隐藏层共256个神经元      
                activation='relu'))                            #定义激活函数ReLU
model2.add(Dropout(0.35))

model2.add(Dense(units=1,                                      #隐藏层共1个神经元，输出1正面评价，输出0负面评价     
                activation='sigmoid'))                         #定义激活函数Sigmoid
  
model2.summary()                                              #查看模型摘要

model2.compile(loss='binary_crossentropy',           
              optimizer='adam',
              metrics=['accuracy'])

train_history2 = model2.fit(x_train,y_train,batch_size=100,    #每一批次100项数据
                            epochs=10,verbose=2,               #执行10个训练周期，verbose=2显示训练过程
                            validation_split=0.2)
scores2 = model2.evaluate(x_test, y_test, verbose=1)



'''
LSTM模型
'''
from keras.layers.recurrent import LSTM

model3 = Sequential()                           
model3.add(Embedding(output_dim=32,                           #将数字列表转换为32维向量
                    input_dim=2000,                           #即单词字典长度
                    input_length=100))                        #即maxlen
model3.add(Dropout(0.35))                                     #每次训练迭代随机放弃20%神经元，以避免过拟合

model3.add(LSTM(32))                                          #LSTM层

model3.add(Dense(units=256,                                  #隐藏层共256个神经元      
                activation='relu'))                          #定义激活函数ReLU
model3.add(Dropout(0.35))

model3.add(Dense(units=1,                                    #隐藏层共1个神经元，输出1正面评价，输出0负面评价     
                activation='sigmoid'))                       #定义激活函数Sigmoid

model3.summary()                                             #查看模型摘要

model3.compile(loss='binary_crossentropy',           
              optimizer='adam',
              metrics=['accuracy'])

train_history3 = model3.fit(x_train,y_train,batch_size=100,      #每一批次100项数据
                            epochs=10,verbose=2,                 #执行10个训练周期，verbose=2显示训练过程
                            validation_split=0.2)
scores3 = model3.evaluate(x_test, y_test, verbose=1)