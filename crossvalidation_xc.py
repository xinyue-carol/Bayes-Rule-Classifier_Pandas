
#data shuffel& cross validation:
import numpy as np
import pandas as pd
from random import seed
from random import random

datadf=pd.read_csv('datadf_midterm_01.csv') 

print(datadf.head())

#shuffel the data set:
datadf_0=datadf.sample(frac=1).reset_index(drop=True)
datadf_0=datadf[['X1','X2','X3','X4','X5','class']]
print(datadf_0.head())


k=5
rate=1/k
length=len(datadf_0)
for z in range(k):
    test_df=datadf_0.iloc[int(z*rate*length):int((z+1)*rate*length),:]
    train_p1=datadf_0.iloc[0:int(z*rate*length),:]
    train_p2=datadf_0.iloc[int((z+1)*rate*length):length,:]
    train_df=pd.concat([train_p1,train_p2],axis=0)

    #separare by class in training set
    strain_df=train_df.sort_values(by=['class'])
    #print(strain_df.head())
    #summarizing data by class
    datalength=strain_df.groupby('class').count()
    #print(datalength)
    class1_df=strain_df.groupby('class').get_group(1)
    class2_df=strain_df.groupby('class').get_group(2)
    class3_df=strain_df.groupby('class').get_group(3)
    class1_list=class1_df.drop(['class'],axis=1).values.tolist()
    class2_list=class2_df.drop(['class'],axis=1).values.tolist()
    class3_list=class3_df.drop(['class'],axis=1).values.tolist()

    #caculating the probabilities ,p(c)
    p_c1=float(len(class1_df))/float(len(train_df))
    p_c2=float(len(class2_df))/(float(len(train_df)))
    p_c3=float(len(class3_df))/(float(len(train_df)))
    #print('p(c1)=',p_c1)
    #print('p(c2)=',p_c2)
    #print('p(c3)=',p_c3)

    def find_pdc_df(theclass):
        from collections import Counter
        p_d_c=Counter([tuple(i) for i in theclass])
        p_d_c_df=pd.DataFrame.from_dict(p_d_c, orient='index').reset_index()
        p_d_c_df=p_d_c_df.rename(columns={'index':'d',0:'count'})
        p_d_c_df['%']=p_d_c_df['count']/p_d_c_df['count'].sum()
        return p_d_c_df

    #caculation the conditional probabilites, p(d|ci)
    #check the calculation by printe out the total sum of the probabiliies
    #calculate p(d|c0):
    pdc_1=find_pdc_df(class1_list)
    pdc_2=find_pdc_df(class2_list)
    pdc_3=find_pdc_df(class3_list)
    #pdc_total=find_pdc_df(class_list)
    #print(len(pdc_1))
    #print(len(pdc_2))
    #print(len(pdc_3))
    #print(len(pdc_total))
    
    #assign the class to the testing set:
    #follow the equations: P(Ci|d)=P(d|Ci)*P(Ci)/P(d)

    #change testing set into a list:
    aclass_list=[]
    td_list=test_df.drop(['class'],axis=1).values.tolist()
    #print(td_list)
    #open the economic gain date:
    eg=pd.read_csv(r"ec.csv")
    #print(eg)
    e_1=float(eg.loc[0,'a_c1'])
    e_2=float(eg.loc[1,'a_c2'])
    e_3=float(eg.loc[2,'a_c3'])
    #print(e_1)
    #print(e_2)
    #print(e_3)
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
    #add the assigned class list into the data frame
    test_df1=test_df
    test_df1['a_class']=aclass_list
    #print(test_df1.head())

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
    
    #output 3,
    #calculate the ecnomic gain:
    eg_list=eg.drop(columns=['Unnamed: 0'],axis=1).values.tolist()
    #print(eg_list)
    a=0
    for i in range(len(matrix_per)):
        for j in range(len(eg_list[i])):
            a=matrix_per[i][j]*float(eg_list[i][j])+a
    print('in fold',str(z),'ecnomic gain is:',a)

