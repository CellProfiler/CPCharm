import numpy as np
import sys

class Classifier(object):
   
   def train(self, labels, data):
      '''Train the classifier'''
      raise NotImplementedError( "train: Should have implemented this" )

   def predict(self, data):
      '''Return a vector containing predicted labels'''
      raise NotImplementedError( "predict: Should have implemented this" )

def normalize(arr):
   arr=np.array(arr, dtype=float)
   max_val=float(arr.max())
   min_val=float(arr.min())
   if max_val!=min_val:
      res=np.array(100.0*((arr-min_val)/(max_val-min_val)))
   else:
      res=np.zeros([len(arr)])
   return res

def wnd_charm_similarity(x, w, c): 
   # See TrainingSet::classify2 in Goldberg's code
   # x: feature vector, [feature_1, feature_2, ..., feature_n]
   # c: class containing vectors from training set, [(feature_1, feature_2, ..., feature_n), ..., (feature_1, feature_2, ..., feature_n)]
   # w: vector of weights for each feature, [weight_1, weight_2, ..., weight_n]
   if len(c)<1:
      print 'ERROR: Empty class'
      sys.exit()
   if len(c.shape)<=1:
      print 'ERROR: Only one sample in class'
      sys.exit()     
   
   p=-5.0
   sim=np.array([])
   wnd=0.0
   
   for t in range(0, len(c)): 
      c_t=np.array(c[t,:], dtype=float)
      dist=np.sum(np.multiply(np.power(w, 2),np.power(x-c_t, 2)))
      sim=np.append(sim, np.power(dist, p))
      
   wnd=np.sum(sim)
   return wnd/len(c)

class WNDCHARMClassifier(Classifier):
   '''
   - the default 'metric' used is the opposite of the cosine similarirty
   - the defaut value of K is 1
   '''

   def __init__(self, distance=wnd_charm_similarity):
      self.distance = distance
      
   def get_class_vectors(self, c):
      t_c=np.array([], dtype='S100')
      first_c=True
      for i in range(0, len(self.labels)):
         if self.labels[i] == c:
            if first_c:
               t_c=np.append(t_c, self.data[i,:])
               first_c=False
            else:
               t_c=np.vstack((t_c, self.data[i,:])) #[(feature_1, ..., feature_n), ..., (feature_1, ..., feature_n)]
      return t_c
   
   def wnd_charm_weight(self, f):
      # See TrainingSet::SetFisherScores on Goldberg's code
      # f: feature vector, [feature_f, ..., feature_f]
      # f must be created as an np.array
      N=len(self.classes)
      
      features_f=np.array(self.data[:,f], dtype=float)
      # Try: if np.sum(features_f)==0
      if np.sum((features_f==False)*1.0)==len(features_f):
         return 0.0
      
      classmean=np.array([]) # mean of feature f within class c
      classvar=np.array([]) # variance of feature f within class c
   
      for c in range(0,N):
         t_c=np.array(self.get_class_vectors(self.classes[c]), dtype=float)
         if len(t_c.shape)<=1:
            print 'ERROR: Only one sample in class'
            sys.exit()            
         temp_mean=np.mean(t_c[:,f])
         classmean=np.append(classmean, temp_mean)
         classvar=np.append(classvar, np.std(t_c[:,f]))
      
      if N>1:
         totalvar=np.var(classmean, ddof=1)
      else:
         totalvar=0.0
         
      meanvar=np.mean(classvar)
      
      if meanvar!=0:
         weight=totalvar/meanvar
      else:
         weight=0.0      
      return weight
  
   def train(self, labels, data):
      self.labels=np.array(labels) # [class, ..., class]
      self.classes=np.array(np.unique(labels))
      self.data=np.array(data)
      
      for f in range(0, len(self.data[0])):
         self.data[:,f]=normalize(self.data[:,f])      

      self.weight = np.array([]) 

      for f in range(0, len(data[0])):
         self.weight=np.append(self.weight,self.wnd_charm_weight(f))

      percent=0.15 # See WND-CHARM paper: 15% of best features used for classification
      temp=np.sort(self.weight)
      index=len(self.weight)-np.round(percent*len(self.weight))
   
      if index>=len(self.weight):
         print 'Warning: Dataset contains too few information'
         index=len(self.weight)-1
      
      thresh=temp[index]
      
      for f in range(0, len(self.weight)):
         if self.weight[f]<thresh:
            self.weight[f]=0.0   

   def predict(self, data):
      '''Return a vector containing predicted labels'''
      data  = np.array(data, dtype=float) 
      predicted_labels = []

      for f in range(0, len(data[0])):
         data[:,f]=normalize(data[:,f])
      
      for x in data:
         predicted_labels.append(self._WND(x))
         
      return np.array(predicted_labels)
         
   def _WND(self, x):
      sim = []
      
      for c in range(0,len(self.classes)): 
         sim.append(wnd_charm_similarity(x, self.weight, self.get_class_vectors(self.classes[c])))

      sim=zip(sim,self.classes)
      return max(sim)[1]




