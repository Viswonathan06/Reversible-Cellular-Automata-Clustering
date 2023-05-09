
import pandas as pd
import numpy as np
from pandas import DataFrame as df 
import math, sys, os
import random as rand
from itertools import combinations, permutations
import pickle as pkl
import threading 
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
from sklearn.cluster import KMeans, Birch
from sklearn import metrics         
import copy 
from numpy import random
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from multiprocessing import Process
from tabulate import tabulateimage.png
import warnings
from bin.BiNCE_encoding import BiNCE
from bin.Frequency_based_encoding import freq_encoding

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
class RCAC:
  def __init__(self,Dataset_name,split_size,num_clusters,data_drop_columns, compare_with_others = False, trials = 10,num_threads = 4) -> None:
    ## Make changes only in this section. Preferably don't change path
    ## The code will automatically make the directories if they don't exist.

    self.save_location = '.temp/'
    self.trials = trials
    self.num_threads = num_threads
    self.labels_ = []
    self.best_so_far = 0
    self.Dataset_name = Dataset_name
    self.rule_kind = 'best'
    self.split_size = split_size
    self.num_clusters = num_clusters
    self.split_index = 0
    self.compare = compare_with_others
    self.data_drop_columns = data_drop_columns
    warnings.filterwarnings("ignore")

    

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  def split_string(self, encodings, div):
    r_ind=[]
    enc_length = len(encodings[0])
    # div = math.floor((enc_length)/split)+1
    for i in range(div, enc_length, div):
      r_ind.append(i)
    iclust=[]
    for i in range(len(encodings)):
      s=0
      for j in r_ind:
        iclust.append(encodings[i][s:j])
        s=j
      iclust.append(encodings[i][s:])
      encodings[i]=iclust
      iclust=[]
    return encodings, len(r_ind)+1 

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  #Function to add null boundary and returns string
  def nullbound(self, n,winsize,init_clusters_):
    if n%2!=0:
            if winsize%2!=0:
              init_clusters_ = "0"*(winsize//2) + init_clusters_ + "0"*(winsize//2)
            else:
              init_clusters_ = "0"*((winsize//2)+1) + init_clusters_ + "0"*(winsize//2)
    else:
            if winsize%2!=0:
              init_clusters_ = "0"*(winsize//2) + init_clusters_ + "0"*(winsize//2)
            else:
              init_clusters_ = "0"*((winsize//2)+1) + init_clusters_ + "0"*(winsize//2)
    return init_clusters_

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  #generating binary rule
  def rg(self, rule, winsize):
    brule=bin(rule).replace("0b","").zfill(2**winsize)
    brule=brule[::-1]
    return brule 

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  def get_rule_list(self, path="./rule_list.txt"):
      # check if best rules exist
      best_path = './config/'+self.Dataset_name+'/Custers-'+str(self.num_clusters)+'/'+self.rule_kind+'_rules.txt'
      try:
        print("Found best configuration!")
        my_file = open(best_path, 'r')
        file_con = my_file.read()
        # replacing end splitting the text 
        # when newline ('\n') is seen.
        rule_list = file_con.split("\n")
        my_file.close()
        self.num_threads = 1
        self.trials = 2
        return rule_list
      except:
        print("No best rules yet. Running trials...")
      # opening the file in read mode
      my_file = open(path, "r")
      # reading the file
      file_con = my_file.read()
      # replacing end splitting the text 
      # when newline ('\n') is seen.
      rule_list = file_con.split("\n")
      my_file.close()
      return rule_list

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  #For finding the length of the next encoding(as in 2-bit, 3-bit encoding)
  def findn(self, d):
      i=1
      while 2**i<d:
        i+=1
      return i

  def cy_enc(self, l,index):
      return bin(index).replace("0b","").zfill(l)

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  #Will apply the CA rule and generate all possible cycles and returns nested list
  def apply_rule(self, split,winsize,brule):
    final_array = []
    split_list=list(set(split))
    split_list.sort(reverse=True)
    split_len=len(split_list[0])
    current_array=[]
    while(split_list):
      curr_element=split_list[0]
      flag=0
      while(not flag):
        if current_array == []:
          current_array.append(curr_element)
          split_list.remove(curr_element)
          
        else:
          curr_element=self.nullbound(split_len,winsize,curr_element)
          t2=""
          for j in range(split_len):
            check=int(str(curr_element[j:j+winsize]),2)
            t2+=brule[check]

          curr_element=t2
          if curr_element not in current_array:
              if curr_element in split_list:
                split_list.remove(curr_element)
                current_array.append(curr_element)
          else:
              flag=1
          
      final_array.append(current_array)
      current_array=[]

    return final_array

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  def aggregate_scores(self):
    path = './'+self.save_location+self.Dataset_name+'/Custers-'+str(self.num_clusters)+'/'
    main_df = pd.DataFrame()
    for files in os.listdir(path):
        if '.DS_Store' in files or '.csv' not in files or 'final_scores' in files:
            continue
        df = pd.read_csv(path+files,index_col=False)
        main_df = main_df.append(df)
    main_df.columns = ["Index","Rule 1", "Rule 2", "Initial clusters", "CA Silhoutte","Heir Silhoutte", "Kmeans Silhoutte", "Birch Silhoutte"]
    main_df = main_df.sort_values('CA Silhoutte', ascending=False)
    main_df.reset_index(inplace = True, drop=True)
    best_score = main_df['CA Silhoutte'].iloc[0]
    best_rules = [main_df['Rule 1'].iloc[0], main_df['Rule 2'].iloc[0]]
    main_df.to_csv('./'+self.save_location+'/'+self.Dataset_name+'/Custers-'+str(self.num_clusters)+'/'+self.rule_kind+'_final_scores.csv')
    return best_score, best_rules

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


  def cellular_automata_clustering(self, split, rules_comb, encoding, thread_no, trials=-1, window_size = 5):
    better_score_list = []
    best_CA_sill = -10000
    best_rule = []
    output_data = []
    columns = ["Rule 1", "Rule 2", "Initial clusters", "CA Silhoutte","Heir Silhoutte", "Kmeans Silhoutte", "Birch Silhoutte"]
    if trials == -1:
      trials = len(rules_comb)
    trial=0
    for rule_set in rules_comb:
      tmc=[]
      tma=[]
      print("Trial number :", trial)
      trial = trial+1
      
      if trial == trials:
        return
      
      print(rule_set)
      enc1 = copy.deepcopy(encoding)
      for p in range(int(len(rule_set)/2)):
        rule = self.rg(int(rule_set[0]), window_size)
        fc = {}
        tr = []
        for i in range(split):
          for j in range(len(enc1)):
            tr.append(enc1[j][i])
          fc[i]=self.apply_rule(tr,window_size,rule)
          tr = []
        # print("Step 0.5 done")
        for i in range(split):
          iclust=fc[i]
          s=self.findn(len(iclust))
          for j in range(len(enc1)):
            plt=enc1[j][i]
            l=0
            while(plt not in iclust[l]):
              l+=1
            enc1[j][i]= self.cy_enc(s,l)
        rule=self.rg(int(rule_set[1]),window_size)
        for i in range(len(enc1)):
          iclust=""
          for j in enc1[i]:
            iclust+=j
          enc1[i]=iclust
        init_clusters=self.apply_rule(enc1,window_size,rule)

        if len(init_clusters)<self.num_clusters:
          return
        enc_data=copy.deepcopy(enc1)
        for i in range(len(enc1)):
          j=0
          while j <len(init_clusters):
            if enc1[i] in init_clusters[j]:
              enc1[i]=j
              break
            else:
              j+=1
        # print("done")
        X=self.data.to_numpy()
        # print(p)
        
        if (len(set(enc1))>1):
          CA_sill_old=silhouette_score(X,enc1,metric="euclidean")
          # print(CA_sill_old)
          clusterer = AgglomerativeClustering(n_clusters=len(set(enc1)), affinity='euclidean', linkage='ward')
          cluster_labels = clusterer.fit_predict(X)
          Heir_sill_old = silhouette_score(X, cluster_labels)

          clust=KMeans(n_clusters=len(set(enc1)),random_state=42)
          clust.fit(X)
          labels=clust.labels_
          
          Kmean_sill_old=metrics.silhouette_score(X, labels, metric='euclidean')
          tmc.append([CA_sill_old,Heir_sill_old])
          if tma==[] or tma[0][0]<CA_sill_old:
            tma.append([CA_sill_old,Heir_sill_old])

      enc_data_=copy.deepcopy(enc_data)
      init_clusters_=copy.deepcopy(init_clusters) 
      for i in range(len(init_clusters)-self.num_clusters):
        enc_data=copy.deepcopy(enc_data_)  
        init_clusters__=copy.deepcopy(init_clusters_)
        idx=0
        for j in range(len(init_clusters__)):
          if len(init_clusters__[j])<len(init_clusters__[idx]):
            idx=j
            
        iclust=copy.deepcopy(init_clusters_) 
        removed=iclust.pop(idx)
        ml=[]

        for j in removed:
          ml.append([enc_data_.index(j),j])
        ml=sorted(ml)
        for j in ml:
          enc_data.remove(j[1])
        #print(ml)
        for j in range(len(init_clusters__[idx])):
          p=[]
          split_list,b=ml[0]
          enc_data.insert(split_list,b)
          X=self.data.copy(deep=True)
          if len(ml)>1:
              for h in ml[1:]:
                X.drop(h[0],inplace=True)
            #print(len(X))
          X=X.to_numpy()
          for k in range(len(iclust)):
            m=copy.deepcopy(iclust)
            m[k].append(b)

            kle=copy.deepcopy(enc_data)
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
          iclust[p.index(max(p))].append(init_clusters__[idx][j])
          ml.pop()
          init_clusters_=iclust
          l=[]
          for i in init_clusters_:
            l.append(len(i))
          #print(l)
      for i in range(len(enc_data_)):
        j=0
        while j <len(init_clusters_):
          if enc_data_[i] in init_clusters_[j]:
            enc_data_[i]=j
            break
          else:
            j+=1
      X=self.data.to_numpy()
      try :
        CA_sill_new=silhouette_score(X,enc_data_,metric="euclidean")
      except:
        CA_sill_new = 0
      
      #### ADD OTHER CLUSTERING METHODS FOR COMPARISON HERE ########

      ## Heirarchical
      clusterer = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='euclidean', linkage='ward')
      cluster_labels = clusterer.fit_predict(X)
      try:
        Heir_sill_new = silhouette_score(X, cluster_labels)
      except:
        Heir_sill_new = 0


      ## Kmeans
      clust=KMeans(n_clusters=self.num_clusters,random_state=42)
      clust.fit(X)
      Km_labels=clust.labels_
      try:
        Kmeans_sill_new=metrics.silhouette_score(X, Km_labels, metric='euclidean')
      except:
        Kmeans_sill_new = 0

      ## Birch 
      clust_model = Birch(n_clusters=self.num_clusters,branching_factor=1500,threshold=1.5)
      clust_model.fit(X)
      labels = clust_model.labels_
      try:
        Birch_new=metrics.silhouette_score(X, labels, metric='euclidean')
      except Exception as e: 
        # print("Error:",e)
        Birch_new = 0

      
      
      # Add rto output data
      output_data.append([rule_set[0],rule_set[1],len(init_clusters),CA_sill_new,Heir_sill_new, Kmeans_sill_new, Birch_new]) 
      out_df = pd.DataFrame(data=output_data, columns=columns)
      out_df.to_csv('./'+self.save_location+'/'+self.Dataset_name+'/Custers-'+str(self.num_clusters)+'/best_'+str(self.split_index)+'_tr_'+str(thread_no)+'.csv')
      
      best_score, best_rules  = self.aggregate_scores()
      if self.best_so_far < best_score:
        self.best_so_far = best_score
        print("Best silhoutte score :", self.best_so_far)
        self.labels_ = enc_data_
        with open('./config/'+self.Dataset_name+'/Custers-'+str(self.num_clusters)+'/'+self.rule_kind+'_rules.txt', 'w') as file1:
          file1.write(str(best_rules[0])+"\n"+str(best_rules[1]))

      if self.compare:
        my_data=[[self.num_clusters,self.best_so_far,Heir_sill_new, Kmeans_sill_new, Birch_new]]
        head=["Final no.of clusters"," Our silhoutte score","Heirarchical", "Kmeans", "Birch"]
        print(tabulate(my_data, headers=head, tablefmt="grid"))
      p=[]
      for i in init_clusters:
        p.append(len(i))
      q=[]
      for i in init_clusters_:
        q.append(len(i))
      

      print("---------------------***-----------------******----------------------***--------------------")

    return best_rule, best_CA_sill, better_score_list




  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

  # Main function
  def fit(self):
    rule_list_name = self.rule_kind+'_cycles_'+str(self.split_size)
    os.makedirs('./'+self.save_location+self.Dataset_name+'/Custers-'+str(self.num_clusters)+'/Final Clusters', exist_ok=True)
    os.makedirs('./config/'+self.Dataset_name+'/Custers-'+str(self.num_clusters), exist_ok=True)
    
    self.data=pd.read_csv("./data/"+self.Dataset_name+".csv")
    self.data=self.data.dropna()
    self.data=self.data.drop(self.data_drop_columns,axis=1)
    window_size = 5
    encoder = BiNCE(self.data, length_of_each=2)
    self.enc = encoder.encode()
    # if you want to apply freq based encoding, 
    # self.enc = freq_encoding(self.data)
    self.enc = list(self.enc[self.enc.columns[0]])
    print(self.enc)
    
    split_enc, num_of_splits = self.split_string(self.enc,self.split_size)
    rule_list = self.get_rule_list('./rules/'+rule_list_name+'.txt')
    rules_comb = list(combinations(rule_list, 2))
    
    thread_pool = []
    rules_comb = np.array(rules_comb)
    rules_comb = rules_comb[:len(rules_comb)]
    rules_sep = np.array_split(rules_comb,(self.num_threads))
    for iclust in range(self.num_threads):
      print('Thread : ',iclust)
      init_clusters_ = threading.Thread(target=self.cellular_automata_clustering, args=(num_of_splits, rules_sep[iclust], split_enc, iclust, self.trials, window_size))
      init_clusters_.daemon = True
      init_clusters_.start()
      thread_pool.append(init_clusters_)
    
    for iclust in thread_pool:
      iclust.join()
    
  



    
  


