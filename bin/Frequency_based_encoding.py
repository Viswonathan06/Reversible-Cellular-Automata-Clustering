import math
import pandas as pd
def freq_encoding(data):
    encoder=['00','01','11','10']
    details= dict()
    for j in data.columns:
        col=data[j]
        sortx = dict()
        a=list()
        for i in col:
            if i not in sortx:
                sortx[i] = 1
                a.append(i)
            else:
                sortx[i]+=1
        a.sort()
        p=0
        s=0
        i=0
        l=0
        while p<len(data):
            m=int(sortx[a[l]])
            if  s+m <=(math.ceil(len(data)/len(encoder))) :
                s+=m
                sortx[a[l]]=encoder[i]
            else:
                if l == 0:
                    sortx[a[l]]=encoder[i]
                    i+=1
                else:
                    i+=1
                    sortx[a[l]]=encoder[i]
                s=0
            l+=1
            p+=m
        details[j]=sortx
        data1 = df(columns = data.columns)
        for i in range(len(data)):
            t=[]
            p=data.loc[i]
            for j in range(len(p)):
                s=details[data.columns[j]]
                t.append(s[p[j]])
            ar = pd.Series(t, index = data1.columns)
            data1 = data1.append(ar,ignore_index = True)

    return data1
