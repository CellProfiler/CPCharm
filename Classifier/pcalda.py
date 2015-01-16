import sklearn.lda as lda
from sklearn.decomposition import PCA
from sklearn.preprocessing import Scaler
import numpy as np

N_COMPONENTS=0.98

class Classifier(object):
   
   def train(self, labels, data):
      '''Train the classifier'''
      raise NotImplementedError( "train: Should have implemented this" )

   def predict(self, data):
      '''Return a vector containing predicted labels'''
      raise NotImplementedError( "predict: Should have implemented this" )

class PCALDAClassifier(Classifier):
   def normalize(self, data, n=N_COMPONENTS):
      X=np.array(data, dtype='float')
      #X=np.array(X[:,np.std(X,0)!=0.0], dtype='float')
      scaler=Scaler()
      Xnorm=scaler.fit_transform(X)
      return Xnorm

   def train(self, labels, data):
      self.labels=np.array(labels) # [class, ..., class]
      self.classes=np.array(np.unique(labels))
      self.data=np.array(data, dtype='float')      
      
      self.ints_labels={}
      for i in range(0, len(self.classes)):
         self.ints_labels[self.classes[i]]=i    
         
      ints=np.zeros([len(self.labels)])
      for i in range(0, len(self.labels)):
         ints[i]=self.ints_labels[self.labels[i]]     
      
      Xnorm=self.normalize(self.data)
      self.pca=PCA(n_components=N_COMPONENTS)
      self.pca.fit(Xnorm)      
      self.pca_data=self.pca.transform(Xnorm)      
      
      self.classifier=lda.LDA()
      self.classifier.fit(self.pca_data, ints) 
      
   def predict(self, data):
      data=np.array(data, dtype=float)
      Xnorm=self.normalize(data)
      pca_data=self.pca.transform(Xnorm)
      
      pred_ints = self.classifier.predict(pca_data)
      pred_labels=np.array([])
      for i in range(0, len(pred_ints)):
         pred_labels=np.append(pred_labels, [key for key, value in self.ints_labels.iteritems() if value == pred_ints[i]][0])
      
      return pred_labels
