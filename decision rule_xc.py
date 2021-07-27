#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from random import random#generate a on dimentional cumulative prbabilites:
#y is the amount of values for the x, and z is the number that the generator will create
def xgenerator(y,z):
    d1_per=[]
    d1_list=np.random.rand(y)
    #print(d1_list)
    a=0
    total=sum(d1_list)
    #normalize the probabilities:
    for i in range(len(d1_list)):
        a+=(d1_list[i]/total)
        d1_per.append(a)
    #print(d1_per)
    xm=[]
    for i in range(z):
        x=np.random.rand() 
        for j in range(y):
            if x<d1_per[j]:
                m=j+1
                xm.append(m)
                break
            elif x>d1_per[j]:
                continue
    #print('there is totally',len(xm),'value in this list')
    return xm 


# In[37]:


x1=xgenerator(2,10000)
x2=xgenerator(2,10000)
x3=xgenerator(2,10000)
x4=xgenerator(2,10000)
x5=xgenerator(2,10000)
c1=xgenerator(3,10000)


# In[88]:


datadf=pd.DataFrame({'X1': x1,'X2': x2,'X3': x3,'X4': x4,'X5': x5,'class':c1})
print(datadf.head())
datadf.to_csv('datadf_midterm.csv')
#data shuffle:
#datadf_0=datadf.sample(frac=1).reset_index(drop=True)
#print(datadf_0.head())


# In[78]:


train_df=datadf[0:int(0.60*float(len(datadf)))]
test_df=datadf[int(0.60*len(datadf)):int(len(datadf))]
print(len(train_df))
print(len(test_df))


# In[79]:


#separare by class in training set
strain_df=train_df.sort_values(by=['class'])
#print(strain_df)
#summarizing data by class
datalength=strain_df.groupby('class').count()
#print(datalength)
class1_df=strain_df.groupby('class').get_group(1)
class2_df=strain_df.groupby('class').get_group(2)
class3_df=strain_df.groupby('class').get_group(3)
class1_list=class1_df.drop(['class'],axis=1).values.tolist()
class2_list=class2_df.drop(['class'],axis=1).values.tolist()
class3_list=class3_df.drop(['class'],axis=1).values.tolist()


# In[80]:


#caculating the probabilities ,p(c)
p_c1=float(len(class1_df))/float(len(train_df))
p_c2=float(len(class2_df))/(float(len(train_df)))
p_c3=float(len(class3_df))/(float(len(train_df)))
print('p(c1)=',p_c1)
print('p(c2)=',p_c2)
print('p(c3)=',p_c3)


# In[81]:


#define the funtion to caculation the conditional probabilites, p(d|ci)
#check the calculation by printe out the total sum of the probabiliies
#calculate p(d|c0):
def find_pdc_df(theclass):
    from collections import Counter
    p_d_c=Counter([tuple(i) for i in theclass])
    p_d_c_df=pd.DataFrame.from_dict(p_d_c, orient='index').reset_index()
    p_d_c_df=p_d_c_df.rename(columns={'index':'d',0:'count'})
    p_d_c_df['%']=p_d_c_df['count']/p_d_c_df['count'].sum()
    return p_d_c_df


# In[82]:


#caculation the conditional probabilites, p(d|ci)
#check the calculation by printe out the total sum of the probabiliies
#calculate p(d|c0):
pdc_1=find_pdc_df(class1_list)
pdc_2=find_pdc_df(class2_list)
pdc_3=find_pdc_df(class3_list)
#pdc_total=find_pdc_df(class_list)
print(len(pdc_1))
print(len(pdc_2))
print(len(pdc_3))
#print(len(pdc_total))


# In[83]:


#output1: P(c|d)
#assign the class to the testing set:
#follow the equations: P(Ci|d)=P(d|Ci)*P(Ci)/P(d)

#change testing set into a list:
aclass_list=[]
td_list=test_df.drop(['class'],axis=1).values.tolist()
#print(td_list)
#open the economic gain date:
eg=pd.read_csv(r"C:\Users\xinyu\Desktop\STUDY\CIS9650\ec.csv")
#print(eg)
e_1=float(eg.loc[0,'a_c1'])
e_2=float(eg.loc[1,'a_c2'])
e_3=float(eg.loc[2,'a_c3'])
print(e_1)
print(e_2)
print(e_3)
for i in range(len(td_list)):
    a=(td_list[i])
    p1=e_1*float(pdc_1.loc[pdc_1['d']==tuple(a),'%'].values)*p_c1
    p2=e_2*float(pdc_2.loc[pdc_2['d']==tuple(a),'%'].values)*p_c2
    p3=e_3*float(pdc_3.loc[pdc_3['d']==tuple(a),'%'].values)*p_c3
    if p1> p2 and p1>p3:
        aclass_list.append(1)
    elif p2> p3 and p2>p1:
        aclass_list.append(2)
    else:
        aclass_list.append(3)
#print(aclass_list)
#print(len(aclass_list))


# In[84]:


#add the assigned class list into the data frame
test_df1=test_df
test_df1['a_class']=aclass_list
print(test_df1.head())


# In[85]:


# Out put2: create confusion matrix:
t_a=test_df1[['class', 'a_class']]
t_a_list=t_a.values.tolist()
total=len(t_a)
#print(t_a_list[0:10])
matrix_count=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(t_a)):
    if t_a_list[i]==[1,1]:
        matrix_count[0][0]+=1
    if t_a_list[i]==[1,2]:
        matrix_count[0][1]+=1
    if t_a_list[i]==[1,3]:
        matrix_count[0][2]+=1
    if t_a_list[i]==[2,1]:
        matrix_count[1][0]+=1
    if t_a_list[i]==[2,2]:
        matrix_count[1][1]+=1
    if t_a_list[i]==[2,3]:
        matrix_count[1][2]+=1
    if t_a_list[i]==[3,1]:
        matrix_count[2][0]+=1
    if t_a_list[i]==[3,2]:
        matrix_count[2][1]+=1
    if t_a_list[i]==[3,3]:
        matrix_count[2][2]+=1
#print(matrix_count)
for i in range(3):
    for j in range(3):
        matrix_count[i][j]=float(matrix_count[i][j])/float(total)
matrix_per=matrix_count
print(matrix_per)


# In[86]:


#output 3,
#calculate the ecnomic gain:
eg_list=eg.drop(columns=['Unnamed: 0'],axis=1).values.tolist()
#print(eg_list)
a=0
for i in range(len(matrix_per)):
    for j in range(len(eg_list[i])):
        a=matrix_per[i][j]*float(eg_list[i][j])+a
print('ecnomic gain is:',a)


# In[76]:


#optimization: 
eg_d=[]
d_value=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008]
for k in range(len(d_value)):
    d=d_value[k]
    print(d)
    #creat a copy of all data frame for all the classes:
    pdc_1c=pdc_1.copy()
    pdc_2c=pdc_2.copy()
    pdc_3c=pdc_3.copy()

    print(pdc_1c['%'].sum())
    print(pdc_2c['%'].sum())
    print(pdc_3c['%'].sum())
    #t_a_list is the list of the (true clase, assigned class)
    print(t_a_list[0:10])

    #td_list is a list of all the d in the testing set
    #print(td_list[0:10])

    delta=0.01
    #print(pdc0.head())
    for i in range(len(t_a_list)):
        if t_a_list[i][0]!=t_a_list[i][1]:
            if t_a_list[i][0]==1:
                pdc_1c.loc[pdc_1c['d']==tuple(td_list[i]),'%']+=d
            if t_a_list[i][0]==2:
                pdc_2c.loc[pdc_2c['d']==tuple(td_list[i]),'%']+=d
            if t_a_list[i][0]==3:
                pdc_3c.loc[pdc_3c['d']==tuple(td_list[i]),'%']+=d
        else:
            continue


    print(pdc_1c['%'].sum())
    print(pdc_2c['%'].sum())
    print(pdc_3c['%'].sum())

    #nomailize the p(d|c) with new columns n_p:

    pdc_1c['n_p']=(pdc_1c['%']/pdc_1c['%'].sum())
    pdc_2c['n_p']=(pdc_2c['%']/pdc_2c['%'].sum())
    pdc_3c['n_p']=(pdc_3c['%']/pdc_3c['%'].sum())
    #pdc0['n_p']=pdc0['%']/total_c0"""
    
#assgined new assigned class using adjusted p(d|Ci), with the list called aclass2_list:
    aclass2_list=[]
#    print(e_1)
#    print(e_2)
#    print(e_3)
    for i in range(len(td_list)):
        a=(td_list[i])
        p1=e_1*pdc_1c.loc[pdc_1['d']==tuple(a),'n_p'].values*p_c1
        p2=e_2*pdc_2c.loc[pdc_2['d']==tuple(a),'n_p'].values*p_c2
        p3=e_3*pdc_3c.loc[pdc_3['d']==tuple(a),'n_p'].values*p_c3
        if p1> p2 and p1>p3:
            aclass2_list.append(1)
        elif p2>p1 and p2>p3:
            aclass2_list.append(2)
        else:
            aclass2_list.append(3)

    #creat new confusion matrix:
    test_df2=test_df
    test_df2['n_a_class']=aclass2_list
    #print(test_df2)
    ntalist=test_df2[['class', 'n_a_class']].values.tolist()
    #print(ntalist)
    matrix2_count=[[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(t_a)):
        if ntalist[i]==[1,1]:
            matrix2_count[0][0]+=1
        if ntalist[i]==[1,2]:
            matrix2_count[0][1]+=1
        if ntalist[i]==[1,3]:
            matrix2_count[0][2]+=1
        if ntalist[i]==[2,1]:
            matrix2_count[1][0]+=1
        if ntalist[i]==[2,2]:
            matrix2_count[1][1]+=1
        if ntalist[i]==[2,3]:
            matrix2_count[1][2]+=1
        if ntalist[i]==[3,1]:
            matrix2_count[2][0]+=1
        if ntalist[i]==[3,2]:
            matrix2_count[2][1]+=1
        if ntalist[i]==[3,3]:
            matrix2_count[2][2]+=1
    #print(matrix_count)
    for i in range(3):
        for j in range(3):
            matrix2_count[i][j]=float(matrix2_count[i][j])/float(len(ntalist))
    matrix2_per=matrix2_count
    print(matrix2_per)
    
    #output 3,
    #calculate the ecnomic gain:
    eg_list=eg.drop(columns=['Unnamed: 0'],axis=1).values.tolist()
    #print(eg_list)
    b=0
    for i in range(len(matrix2_per)):
        for j in range(len(eg_list[i])):
            b=matrix2_per[i][j]*float(eg_list[i][j])+b
    print('new ecnomic gain is:',b)
    eg_d.append(b)
print(eg_d)


# In[66]:


from matplotlib import pyplot as plt
plt.plot(d_value, eg_d)
plt.show()






