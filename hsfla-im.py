
# coding: utf-8

# In[1]:


import networkx as nx
import csv
import igraph
import random
import numpy as np
from igraph import *
from collections import deque


# In[2]:


"""g = igraph.Graph.Read_Ncol('hep.txt', directed=True)
g=g.as_undirected()"""


# In[3]:


g = Graph.Read_GML("netscience.gml")
g=g.as_undirected()


# In[4]:


neighbors_list = g.get_adjlist(mode=OUT)


# In[5]:


verclus=g.community_multilevel()   # Using multilevel community detection 

total_clusters=len(verclus)

print(verclus)
print(g.modularity(verclus))


# In[6]:


w=0.5
sz = np.zeros((total_clusters))
sz1=np.zeros((total_clusters))
"""for i in range(total_clusters):
    edge=0
    for j in verclus[i]:
        templist=neighbors_list[j]
        for k in templist:
            if k not in verclus[i]:
                edge+=1
    sz1[i]=edge
for i in range(total_clusters):
    sz[i]=w*len(verclus[i])+(1-w)*sz1[i] """
for i in range(total_clusters):
    sz[i]=len(verclus[i])
sort_order = np.argsort(sz)
#print(sort_order)
num_sig = 20
cnt=0
sig_com = []
for i in range(total_clusters):
    sig_com.append(verclus[sort_order[total_clusters-1-i]])
    cnt+=1;
    if cnt==num_sig:
        break
print(len(sig_com))
print(sig_com)


# In[7]:


num_candidates = []

max_c = -1
min_c = 10000000000

for i in range(num_sig):
    if(len(sig_com[i]) < min_c):
        min_c = len(sig_com[i])
    if(len(sig_com[i]) > max_c):
        max_c = len(sig_com[i])
print("Min Max ",min_c,max_c)        
alpha = 4
beta = 10

for i in range(num_sig):
   # print(len(sig_com[i]))
    x = (len(sig_com[i])-min_c)/(max_c-min_c)*beta + alpha
    if x>len(sig_com[i]):
        x=len(sig_com[i])
    num_candidates.append(int(x))
    
print(num_candidates)    
total_candidates=0
for y in num_candidates:
    total_candidates+=y
print(total_candidates)    


# In[8]:


deg = g.indegree()
len(deg)


# In[9]:


def closeness_centrality(particle):
    visit=np.zeros((g.vcount()))
    cc=np.zeros((g.vcount()))
    q=deque()
    q.append(particle)
    cc[particle]=0
    while q:
        cur=q.popleft()
        visit[cur]=1
        for j in neighbors_list[cur]:
            if visit[j]==0:
                cc[j]=cc[cur]+1
                q.append(j)
    return np.sum(cc)


# In[10]:


def LAC(particle):
    list=neighbors_list[particle]
    l=len(list)
    sum=0
    for i in range(l):
        for j in range(i+1,l):
            if j in neighbors_list[i]:
                sum+=1
    if l==0:
        return 0
    return sum/l


# In[11]:


candidates=[]
for i in range(num_sig):
    temp = sig_com[i]
    degs = np.zeros((len(temp)))
    for j in range(len(temp)):
        degs[j]=deg[temp[j]]
        #degs[j]=LAC(temp[j])
        #degs[j]=closeness_centrality(temp[j])
    sort_order=np.argsort(degs)
    #desc_order = np.flip(sort_order,0)
    sz=len(sort_order)
    for z in range(num_candidates[i]):
        candidates.append(sig_com[i][sort_order[sz-1-z]])
        #candidates.append(sig_com[i][sort_order[z]])
print(len(sig_com))
print(candidates)        
print(len(candidates))


# In[12]:


def similarity(u,v):
    neighbors_u = set()
    neighbors_u.add(u)
    for i in neighbors_list[u]:
        neighbors_u.add(i)
    neighbors_v = set()
    neighbors_v.add(v)
    for i in neighbors_list[v]:
        neighbors_v.add(i)
    sim = len(neighbors_u.intersection(neighbors_v))/(len(neighbors_u) + len(neighbors_v))
    return sim


# In[13]:


random.seed(200)
def SHD(candidates,sim):
    x = []
    temp_candidate = set()
    for xx in candidates:
        temp_candidate.add(xx)
    for i in range(num_seeds):
        max_degree = 0
        for node in temp_candidate:
            if deg[node] > max_degree:
                max_degree = deg[node]
                v = node
        if len(x)<num_seeds:
            x.append(v)
        neighbors = neighbors_list[v]
        sim_neighbors = set()
        for neighbor in neighbors:
            if similarity(v,neighbor) > sim:
                sim_neighbors.add(neighbor)
        temp_candidate.discard(v)
        for sim_neighbor in sim_neighbors:
            temp_candidate.discard(sim_neighbor)
        if len(temp_candidate) == 0: 
            remaining = num_seeds-1-i
            rem=0
            while len(x)<num_seeds:
                vertex = random.choice(candidates)
                if vertex in x:
                    continue
                x.append(vertex)    
                rem+=1    
    return x            


# In[14]:


num_seeds = 30 #SEEDS
sim = 0.1
seeds = SHD(candidates,sim)


# In[15]:


def find_fitness(population):
  p = 0.05
  fitness = np.zeros((population.shape[0]))
  idx = 0  
  for frog in population:
      term1 = num_seeds
      for i in frog:
        neighbors = neighbors_list[i]
        term1 += deg[i]*p
        for j in neighbors:
          term1 += deg[j]*p*p

      term2 = 0
      for i in frog:
        neighbors = neighbors_list[i]
        temp = 0
        for j in neighbors:
          if j in frog:
              temp = temp + p * (deg[j]*p - p)
        term2 = term2 + temp

      term3 = 0
      for i in (frog):
        temp1 = 0
        neighbors = neighbors_list[i]
        for c in neighbors:
            temp2=0
            if c in frog:
                sec_neighbors = neighbors_list[c]
                for d in sec_neighbors:
                    if d in frog and d != i:
                       temp2 = temp2 + p * p
            temp1 = temp1 + temp2
        term3=term3+temp1

      fitness[idx]= (term1-term2-term3)
      idx+=1
  return fitness      


# In[16]:


def find_fitness_single(frog):
      p=0.05
      fitness=0
      term1 = num_seeds
      for i in frog:
        neighbors = neighbors_list[i]
        term1 += deg[i]*p
        for j in neighbors:
          term1 += deg[j]*p*p

      term2 = 0
      for i in frog:
        neighbors = neighbors_list[i]
        temp = 0
        for j in neighbors:
          if j in frog:
              temp = temp + p * (deg[j]*p - p)
        term2 = term2 + temp

      term3 = 0
      for i in frog:
        temp1 = 0
        neighbors = neighbors_list[i]
        for c in neighbors:
            temp2=0
            if c in frog:
                sec_neighbors = neighbors_list[c]
                for d in sec_neighbors:
                    if d in frog and d != i:
                       temp2 = temp2 + p * p
            temp1 = temp1 + temp2
        term3=term3+temp1

      fitness = (term1-term2-term3)
      
      return fitness


# In[17]:


pop_size = 100
max_iterations = 200
alpha = 0.2
population = np.zeros((pop_size,num_seeds),dtype=int)

#step 0
n = 4   #number of frogs in each memeplex
m = 25   #number of memeplexes
#step 1
fit = np.zeros((pop_size),dtype=float)
temparray = np.zeros((num_seeds),dtype=int)
temp=0

memeplex = np.zeros((m,n,num_seeds),dtype=int)
px = np.zeros((num_seeds),dtype=int)

"""for i in range(pop_size):
    population[i] = SHD(candidates,0.05*(i+1))"""
shd = SHD(candidates,0.05)
print("Initial fitness: ", find_fitness_single(shd))
for i in range(pop_size//2):
    population[i] = shd.copy()
    available ={}
    for cand in candidates:
        available[cand] = 1
    
    for ss in population[i]:
        if ss in available:
            del available[ss] 
    for j in range(num_seeds):
        if random.random()>=0.5:
                random.seed(i*pop_size//2+j)
            
                
                r = random.choice(list(available.keys()))
                tempo = population[i][j] 
                population[i][j] = r
                del available[r]
                available[tempo]=1



for i in range(pop_size//2,pop_size):
    for cand in candidates:
        available[cand] = 1
    for j in range(num_seeds):
                random.seed(i*pop_size//2+j)
                r = random.choice(list(available.keys()))
             
                population[i][j] = r
                del available[r]
                    
                
"""for frog in population:
    available = {}
    for cand in candidates:
        available[cand]=1
        
    for k in range(num_seeds):
                r = random.choice(list(available.keys()))
                
                frog[k] = r
                del available[r]"""
      
print(population)    


# In[18]:


fit=find_fitness(population)
print(fit)  

order = np.argsort(fit)
order = np.flipud(order)
new_pop = np.zeros((pop_size,num_seeds),dtype=int) 

for it in range(len(order)):
    new_pop[it] = population[order[it]].copy()
for it in range(len(order)):
    population[it] = new_pop[it].copy() 

px=population[0].copy()

#print(swarm_pos)
fit=find_fitness(population)
print(fit) 
print(len(candidates))


# In[19]:


def max_influence(array):
    max=-1
    index=-1
    for i in range(len(array)):
        neighbors=neighbors_list[array[i]]
        sum=0
        for j in neighbors:
            sum=sum+deg[j]
        if sum>max:
            max=sum
            index=i
    return array[index]


# In[ ]:


for itr in range(max_iterations):     ###max iterations
    
    #step 3
    cnt = 0;
    for i in range(n):
        for j in range(m):
            memeplex[j][i] = population[cnt]
            cnt+=1;
            
    #step 4.0,4,1 and 4.2
    max_ls=10
    pb=np.zeros((num_seeds),dtype=int)
    pw=np.zeros((num_seeds),dtype=int)
   
    temparray=np.zeros((num_seeds),dtype=int)
    
 
    for im in range(m):
        for iN in range(max_ls):
            
          #step 4.3
          q = n//2
          pb = memeplex[im][0].copy()
          pw = memeplex[im][q-1].copy()
        
         
          temparray=pw
          min_fit=find_fitness_single(temparray)
            
          #step 4.4
          available = {}
          for cand in candidates:
                        available[cand]=1
          for k in range(num_seeds):
                if pw[k] in pb:
                    #print(pw[k])
                    if pw[k] in available:
                        del available[pw[k]]
                    continue
                #r = random.choice(list(available.keys()))
                r=max_influence(list(available.keys()))
                tempo = pw[k] 
                pw[k] = r
                del available[r]
                available[tempo]=1
                """ = random.randrange(0,len(candidates),1);
                while candidates[r] in pw:
                    r = random.randrange(0,len(candidates),1);
                pw[k] = candidates[r]"""
                
          #step 4.5
          if find_fitness_single(pw) < find_fitness_single(temparray):
            available = {}
            for cand in candidates:
                available[cand] = 1
            pw = temparray.copy()    
            """for k in range(num_seeds):
                if pw[k] in px:
                    continue
                r = random.randrange(0,len(candidates),1);
                while candidates[r] in pw:
                    r = random.randrange(0,len(candidates),1);
                pw[k] = candidates[r]"""
            for k in range(num_seeds):
                if pw[k] in px:
                    if pw[k] in available:
                        del available[pw[k]]
                    continue
                #r = random.choice(list(available.keys()))
                r=max_influence(list(available.keys()))
                tempo = pw[k]
                pw[k] = r
                del available[r]
                available[tempo]=1
                
          #step 4.6
          if find_fitness_single(pw) < min_fit:
            available = {}
            for cand in candidates:
                available[cand] = 1 
            pw1=pw.copy()
            """for k in range(num_seeds):
                r = random.randrange(0,len(candidates),1);
                while candidates[r] in pw:
                    r = random.randrange(0,len(candidates),1);
                pw[k] = candidates[r]"""
            percent=0.2
            numchange=int(num_seeds*percent)
            pos=np.random.randint(0,num_seeds-1,numchange)
            for k in pos:
                #r=random.choice(list(available.keys()))
                r=max_influence(list(available.keys()))
                tempo=pw[k]
                pw[k]=r
                del available[r]
                available[tempo]=1
            if find_fitness_single(pw)> 0.9*find_fitness_single(pw1):
                if find_fitness_single(pw)<=find_fitness_single(pw1):
                    max_worse=15
                    percent=0.2
                    numchange=int(num_seeds*percent)
                    for j in range(max_worse):
                        pos=np.random.randint(0,num_seeds-1,numchange)
                        for k in pos:
                            #r=random.choice(list(available.keys()))
                            r=max_influence(list(available.keys()))
                            tempo=pw[k]
                            pw[k]=r
                            del available[r]
                            available[tempo]=1
                        if find_fitness_single(pw)>min_fit:
                            break
                    if j==max_worse:
                        pos=np.random.randint(0,num_seeds-1,numchange)
                        for k in pos:
                            r=random.choice(list(available.keys()))
                            #r=max_influence(list(available.keys()))
                            tempo=pw[k]
                            pw[k]=r
                            del available[r]
                            available[tempo]=1
                elif find_fitness_single(pw)<min_fit:
                    pw=temparray
            else:
                pos=np.random.randint(0,num_seeds-1,numchange)
                for k in pos:
                    r=random.choice(list(available.keys()))
                    #r=max_influence(list(available.keys()))
                    tempo=pw[k]
                    pw[k]=r
                    del available[r]
                    available[tempo]=1
                
          #step 4.7
          memeplex[im][q-1] = pw.copy()
           
         
            
          #sorting them
          fit=find_fitness(memeplex[im])
          order = np.argsort(fit)
          order = np.flipud(order)
          new_pop = np.zeros((n,num_seeds),dtype=int) 
          for it in range(len(order)):
              new_pop[it] = memeplex[im][order[it]].copy()
          for it in range(len(order)):
              memeplex[im][it] = new_pop[it].copy() 
                
        

    #step 5
    cnt = 0 
    for i in range(m):
        for j in range(n):
            population[cnt] = memeplex[i][j].copy()
            cnt += 1
     
    fit=find_fitness(population)
    order = np.argsort(fit)
    order = np.flipud(order)
    new_pop = np.zeros((pop_size,num_seeds),dtype=int) 
    for it in range(len(order)):
        new_pop[it] = population[order[it]].copy()
    for it in range(len(order)):
        population[it] = new_pop[it].copy() 
        
    px=population[0]
    print("#",itr,": ",find_fitness_single(px))


 


# In[ ]:


fitness = find_fitness(population)

# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))[0]
best_match_idx = best_match_idx[0]

best_solution = px

best_solution_fitness = fitness[best_match_idx]

print("best_match_idx : ", best_match_idx)
print("best_solution : ", best_solution)

print("Best solution fitness : ", best_solution_fitness)


# In[ ]:


new_seeds = [14,38,0,7,59,51,29,20,1,57]
xx = (1, 10)
 
# Creating the initial population.
yy = np.random.randint(low=0, high=2, size=xx)
yy[0] = new_seeds

print(yy)
find_fitness(yy)

