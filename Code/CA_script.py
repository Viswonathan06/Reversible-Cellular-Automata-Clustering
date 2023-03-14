# %%
import pandas as pd
import numpy as np
from pandas import DataFrame as df 
import math, sys
import random as rand
from itertools import combinations
import pickle as pkl
import threading 
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn import metrics         
import copy 
from tabulate import tabulate
import random

# %%
# data=pd.read_csv("total_data_na.csv")
data=pd.read_csv("../Dataset/school_district_breakdowns_1.csv")
data=data.dropna()
data=data.drop("jurisdiction_name",axis=1)

# %%
#Frequency Based Encoding
def frequency_encoding(data):
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



# %%
#Generating final encoded string
def get_concatenated_string(data):
  enc=[]
  for i in range(len(data)):
    p=data.loc[i]
    t=""
    for j in p:
      t+=j
    enc.append(t)
  return enc

# %%
def split_string(encodings, split = 5):
  r_ind=[]
  enc_length = len(encodings[0])
  div = math.floor((enc_length)/split)+1
  for i in range(div, enc_length, div):
    r_ind.append(i)
  print("Split index is :")
  print(r_ind, end='\n\n')
  t=[]
  for i in range(len(encodings)):
    s=0
    for j in r_ind:
      t.append(encodings[i][s:j])
      s=j
    t.append(encodings[i][s:])
    encodings[i]=t
    t=[]
  # for i in encodings:
    # print(i)
  return(encodings)  



# %%
#Function to add null boundary and returns string
def nullbound(n,winsize,t1):
  if n%2!=0:
          if winsize%2!=0:
            t1 = "0"*(winsize//2) + t1 + "0"*(winsize//2)
          else:
            t1 = "0"*((winsize//2)+1) + t1 + "0"*(winsize//2)
  else:
          if winsize%2!=0:
            t1 = "0"*(winsize//2) + t1 + "0"*(winsize//2)
          else:
            t1 = "0"*((winsize//2)+1) + t1 + "0"*(winsize//2)
  return t1

# %%
#generating binary rule
def rg(rule, winsize):
  brule=bin(rule).replace("0b","").zfill(2**winsize)
  brule=brule[::-1]
  return brule 

# %%
def get_rule_list(path="../rule_list.txt"):
    # opening the file in read mode
    my_file = open(path, "r")
    # reading the file
    file_con = my_file.read()
    # replacing end splitting the text 
    # when newline ('\n') is seen.
    rule_list = file_con.split("\n")
    my_file.close()
    return rule_list

# %%
#For finding the length of the next encoding(as in 2-bit, 3-bit encoding)
def findn(d):
    i=1
    while 2**i<d:
      i+=1
    return i

def cy_enc(l,index):
    return bin(index).replace("0b","").zfill(l)

# %%
#Will apply the CA rule and generate all possible cycles and returns nested list
def apply_rule(a1,winsize,brule):
  final_array = []
  a=list(set(a1))
  n=len(a[0])
  current_array=[]
  while(a):
    t1=a[0]
    flag=0
    while(not flag):
      if current_array == []:
        current_array.append(t1)
        a.remove(t1)
        
      else:
        t1=nullbound(n,winsize,t1)
        t2=""

        for j in range(n):
          check=int(str(t1[j:j+winsize]),2)
          t2+=brule[check]

        t1=t2

        if t1 not in current_array:
            if t1 in a:
              a.remove(t1)
              current_array.append(t1)
              
        else:
            flag=1
        

    final_array.append(current_array)
    current_array=[]
    cycle=1
    # print(cycle)
    cycle+=1
  return final_array

# %%
def cellular_automata_clustering(split, rules_comb, encoding, thread_no, trials=-1, window_size = 5):

  better_score_list = []
  best_CA_sill = -10000
  best_rule = []
  output_data = []
  columns = ["Rule 1", "Rule 2", "Initial clusters", "CA Silhoutte","Kmeans Silhoutte", "New Clusters","New CA Silhoutte","New Kmeans Silhoutte"]
  print(len(rules_comb))
  if trials == -1:
    trials = len(rules_comb)
  trial=0
  for rule_set in rules_comb:
    tmc=[]
    tma=[]
    print("Trial number :", trial)
    trial = trial+1
    
    if trial == trials:
      break
    
    print(rule_set)
    enc1 = copy.deepcopy(encoding)
    for p in range(int(len(rule_set)/2)):
      rule = rg(int(rule_set[0]), window_size)
      fc = {}
      tr = []
      for i in range(split):
        for j in range(len(enc1)):
          tr.append(enc1[j][i])
        fc[i]=apply_rule(tr,window_size,rule)
        tr = []

      for i in range(split):
        t=fc[i]
        s=findn(len(t))
        for j in range(len(enc1)):
          plt=enc1[j][i]
          l=0
          while(plt not in t[l]):
            l+=1
          enc1[j][i]= cy_enc(s,l)
      rule=rg(int(rule_set[1]),window_size)
      for i in range(len(enc1)):
        t=""
        for j in enc1[i]:
          t+=j
        enc1[i]=t
      # print(len(enc1))
      fa=apply_rule(enc1,window_size,rule)
      #print(fa)
      kl=copy.deepcopy(enc1)
      for i in range(len(enc1)):
        j=0
        while j <len(fa):
          if enc1[i] in fa[j]:
            enc1[i]=j
            break
          else:
            j+=1

      
      X=data.to_numpy()
      #print(len(enc1),len(X))
      print(p)
      
      if (len(set(enc1))>1):
        CA_sill_old=silhouette_score(X,enc1,metric="euclidean")
        print(CA_sill_old)
        # print(rule_set[2*p+1])
        # print("silhoutte:",CA_sill_old)
        # !!! CHange this later
        # if CA_sill_old < best_CA_old:
        #   best_CA_old = CA_sill_old
        #   best_rules= rule_set
        # print("calinski:",calinski_harabasz_score(X,enc1))
        # print("davies:",davies_bouldin_score(X,enc1))

        

        clust=KMeans(n_clusters=len(set(enc1)),random_state=42)
        clust.fit(X)
        labels=clust.labels_
        
        Kmean_sill_old=metrics.silhouette_score(X, labels, metric='euclidean')
        # print("Kmeans silhoutte :" ,Kmeans_sill_new)
        # print(len(set(labels)))
        # print("Kmeans calinski :", (metrics.calinski_harabasz_score(X, labels)))
        # print("Kmeans davies_bouldin_score: ",metrics.davies_bouldin_score(X, labels))
        # if Kmean_sill_old < CA_sill_old:
        #   better_score_list.append([rule_set, CA_sill_old, Kmean_sill_old])
        tmc.append([CA_sill_old,Kmean_sill_old])
        if tma==[] or tma[0][0]<CA_sill_old:
          tma.append([CA_sill_old,Kmean_sill_old])

    kls=copy.deepcopy(kl)
    t1=copy.deepcopy(fa) 
    for i in range(len(fa)-2):
      #print(len(t1),"###")
      kl=copy.deepcopy(kls)  
      fa1=copy.deepcopy(t1)
      idx=0
      for j in range(len(fa1)):
        if len(fa1[j])<len(fa1[idx]):
          idx=j
          
      t=copy.deepcopy(t1) 
      removed=t.pop(idx)
      #print(removed)
      ml=[]

      for j in removed:
        ml.append([kls.index(j),j])
      ml=sorted(ml)
      for j in ml:
        kl.remove(j[1])
      #print(ml)
      for j in range(len(fa1[idx])):
        p=[]
        a,b=ml[0]
        kl.insert(a,b)
        X=data.copy(deep=True)
        if len(ml)>1:
            for h in ml[1:]:
              X.drop(h[0],inplace=True)
          #print(len(X))
        X=X.to_numpy()
        for k in range(len(t)):
          m=copy.deepcopy(t)
          m[k].append(b)

          kle=copy.deepcopy(kl)
          for z in range(len(kle)):
            c=0
            while c <len(m):
              if kle[z] in m[c]:
                kle[z]=c
                break
              else:
                c+=1
          CA_sill_new=silhouette_score(X,kle,metric="euclidean")
          p.append(CA_sill_new)
        #print(p.index(max(p)))
        #print(t[p.index(max(p))])
        t[p.index(max(p))].append(fa1[idx][j])
        ml.pop()
        t1=t
        l=[]
        for i in t1:
          l.append(len(i))
        #print(l)
    for i in range(len(kls)):
      j=0
      while j <len(t1):
        if kls[i] in t1[j]:
          kls[i]=j
          break
        else:
          j+=1
    X=data.to_numpy()
    CA_sill_new=silhouette_score(X,kls,metric="euclidean")
    print("silhoutte:",CA_sill_new)
    print("calinski:",calinski_harabasz_score(X,kls))
    print("davies:",davies_bouldin_score(X,kls))
    clust=KMeans(n_clusters=len(set(kls)),random_state=42)
    clust.fit(X)
    labels=clust.labels_
        
    Kmeans_sill_new=metrics.silhouette_score(X, labels, metric='euclidean')


    if CA_sill_new > Kmeans_sill_new:
      better_score_list.append(rule_set)
    
    if CA_sill_new > best_CA_sill:
      best_CA_sill = CA_sill_new
      best_rule = rule_set


    my_data=[[len(fa),tma[0][0],tma[0][1],CA_sill_new,Kmeans_sill_new]]
    head=["Initial no.of clusters", "Silhoutte score for our algo","silhoutte score for kmeans"," new silhoutte score","new silhoute for kmeans"]
    

    # Add rto output data
    output_data.append([rule_set[0],rule_set[1],len(fa),tma[0][0],tma[0][1],len(t1),CA_sill_new,Kmeans_sill_new]) 
    out_df = pd.DataFrame(data=output_data, columns=columns)
    out_df.to_csv('../Results/Exhaustive Search/Entropy Based Hash/Thread_'+str(thread_no)+'.csv')


    print(tabulate(my_data, headers=head, tablefmt="grid"))
    p=[]
    for i in fa:
      p.append(len(i))
    q=[]
    for i in t1:
      q.append(len(i))
    print("before :",p)
    print("after: ",q)

    print(t1)
    print("---------------------***-----------------******----------------------***--------------------")
  len(output_data)
  # print(out_df.head())
  return best_rule, best_CA_sill, better_score_list

# %%
#Replacing the data in table with encoding
if __name__ == "__main__":
  
  freq_data = frequency_encoding(data)
  enc = get_concatenated_string(freq_data)
  enc = list(pd.read_csv('../Dataset/entropy_hashed_SDBdata.csv')['0'])

  split_enc = split_string(enc, split=5)
  rule_list = get_rule_list()
  rules_comb = list(combinations(rule_list, 2))
  trials = int(sys.argv[2])
  num_threads = int(sys.argv[1])
  thread_pool = []
  rules_comb = np.array(rules_comb)
  print(type(rules_comb))
  rules_sep = np.array_split(rules_comb,(num_threads))
  print(len(rules_sep))
  for t in range(num_threads):
    print('Thread : ',t)

    t1 = threading.Thread(target=cellular_automata_clustering, args=(5, rules_sep[t], split_enc, t, trials,))
    t1.daemon = True
    t1.start()
    thread_pool.append(t1)
  
  for t in thread_pool:
    t.join()

