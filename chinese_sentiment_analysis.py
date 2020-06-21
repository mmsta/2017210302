import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn import svm
from sklearn import naive_bayes
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

'''
中文情感分析
商品（书籍、酒店、计算机、牛奶、手机、热水器）评论情感分析 
'''

#载入数据，做预处理（分词），切分训练集与测试集
def load_file_and_preprocessing():
    neg=pd.read_excel('chinese_review/neg.xls',header=None,index=None)
    pos=pd.read_excel('chinese_review/pos.xls',header=None,index=None)
    cw=lambda x:list(jieba.cut(x))
    pos['words']=pos[0].apply(cw)
    neg['words']=neg[0].apply(cw)
    print(neg['words'])
    print(pos['words'])
    #用标签1表示积极，用0表示消极
    y=np.concatenate((np.ones(len(pos)),np.zeros(len(neg))))

    #训练集：测试集=8:2
    x_train,x_test,y_train,y_test=train_test_split(np.concatenate((pos['words'],neg['words'])),y,test_size=0.2)


    #NumPy提供了多种文件操作函数方便存取数组内容（npy格式以二进制存储数据的）
    np.save('data/y_train.npy',y_train)
    np.save('data/y_test.npy',y_test)
    return x_train,x_test

#对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text,size,w2v_model):
    vec=np.zeros(size).reshape((1,size))
    count=0
    for word in text:
        try:
            vec+=w2v_model[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec

#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim=300 #词向量维度
    #初始化模型和词表
    w2v_model=Word2Vec(size=n_dim,window=5,sg=0,hs=0,negative=5,min_count=10)
    w2v_model.build_vocab(x_train) #准备模型词汇表

    #在评论训练集上建模
    w2v_model.train(x_train,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter) #训练词向量
    #训练集评论向量集合
    train_vecs=np.concatenate([build_sentence_vector(z,n_dim,w2v_model) for z in x_train])
    np.save('data/train_vecs.npy',train_vecs) #将训练集保存到文件中
    print('训练集的维度：',train_vecs.shape)  #输出训练集的维度

    #在测试集上训练
    w2v_model.train(x_test,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter)
    w2v_model.save('data/w2v_model/w2v_model.pkl')
    test_vecs=np.concatenate([build_sentence_vector(z,n_dim,w2v_model) for z in x_test])
    np.save('data/test_vecs.npy',test_vecs)
    print('测试集的维度：',test_vecs.shape)#输出测试集的维度

#获得训练集向量和标签，测试集向量和标签
def get_data():
    train_vecs=np.load('data/train_vecs.npy')
    print('训练集向量为：',train_vecs)
    y_train=np.load('data/y_train.npy')
    print('训练集标签为：',y_train)
    test_vecs=np.load('data/test_vecs.npy')
    print('测试集向量为：',test_vecs)
    y_test=np.load('data/y_test.npy')
    print('测试集标签为：',y_test)
    return train_vecs,y_train,test_vecs,y_test

#定义超平面函数
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=True, 
                    title='hyperplan'):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
        
        
#训练SVM模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    X, y = make_blobs(n_samples=100, centers=2, 
                  random_state=0, cluster_std=0.3)
    clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
    clf_rbf.fit(train_vecs,y_train)   #根据给定的训练数据拟合SVM模型
    joblib.dump(clf_rbf,'data/svm_model/model.pkl')  #保存训练好的SVM模型

    print ('训练集准确率：', accuracy_score(clf_rbf.predict(train_vecs), y_train))
    print ('测试集准确率：', accuracy_score(clf_rbf.predict(test_vecs), y_test))
    
    print ('训练集召回率：', recall_score(clf_rbf.predict(train_vecs), y_train))
    print ('测试集召回率：', recall_score(clf_rbf.predict(test_vecs), y_test))
    
    clf_rbf.fit(X, y)
    plt.figure(figsize=(5,5), dpi=144)
    plot_hyperplane(clf_rbf, X, y, h=0.01, 
                title='Gaussian Kernel with $\gamma=0.5$')

   
#训练朴素贝叶斯模型
def bayes_train(train_vecs,y_train,test_vecs,y_test):
    clf = naive_bayes.GaussianNB()
    clf.fit(train_vecs,y_train)
    joblib.dump(clf,'data/bayes_model/model.pkl')
    print ('训练集准确率：', accuracy_score(clf.predict(train_vecs), y_train))
    print ('测试集准确率：', accuracy_score(clf.predict(test_vecs), y_test))
    
    print ('训练集召回率：', recall_score(clf.predict(train_vecs), y_train))
    print ('测试集召回率：', recall_score(clf.predict(test_vecs), y_test))


#构建待遇测句子的向量
def get_predict_vecs(words):
    n_dim=300
    w2v_model=Word2Vec.load('data/w2v_model/w2v_model.pkl')
    train_vecs=build_sentence_vector(words,n_dim,w2v_model)
    return train_vecs

#基于SVM对单个句子进行情感判断
def svm_predict(string):
    #对要进行预测的评论语句进行预处理
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('data/svm_model/model.pkl')
    result=clf.predict(words_vecs)

    if int(result[0])==1:
        print(string,'\n','SVM预测结果为：positive')
    else:
        print(string,'\n','SVM预测结果为：negative')

#基于朴素贝叶斯对单个句子进行情感判断      
def bayes_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('data/bayes_model/model.pkl')
    result=clf.predict(words_vecs)

    if int(result[0])==1:
        print(string,'\n','朴素贝叶斯模型预测结果为：positive')
    else:
        print(string,'\n','朴素贝叶斯模型预测结果为：negative')



if __name__=='__main__':
    x_train, x_test=load_file_and_preprocessing()
    get_train_vecs(x_train, x_test)

    train_vecs, y_train, test_vecs, y_test=get_data()
    svm_train(train_vecs, y_train, test_vecs, y_test)
    
    train_vecs, y_train, test_vecs, y_test=get_data()
    bayes_train(train_vecs, y_train, test_vecs, y_test)

    #对输入句子情感进行判断
    #string='电池充完了电连手机都打不开，简直烂的要命，真是金玉其外，败絮其中！连5号电池都不如'
    #string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    string='这款电脑感觉还行，能用，先试用一下，应该不错'
    svm_predict(string)
    bayes_predict(string)
