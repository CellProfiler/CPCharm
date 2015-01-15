#! /usr/bin/env python
import numpy as np
import re
import sys
import csv
import pickle
import wndcharm
import pcalda

def openobject(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def format_data(data):
    data=np.array(data)
    #data=data[1:len(data),:]
    ind=-1
    for i in range(0, len(data[0,:])):
        if re.search("[Kk]ey", data[0,i]):
            ind=i
              
    if ind<0:
        print 'ERROR: Image identifiers are absent.'
        sys.exit()
    
    data[:,[0,ind]]=data[:,[ind,0]]
    return np.copy(data)
    
classifier_path=sys.argv[1]
data_path=sys.argv[2]
output_path=sys.argv[3]

temp_data=csv.reader(open(data_path, 'rb'), delimiter=',')
lst_data=[]
for row in temp_data: lst_data.append(row)
data=np.array(lst_data)
data=format_data(data)

data_key=data[1:,0]
data_data=data[1:,1:]

classifier=openobject(classifier_path)

if type(classifier) != wndcharm.WNDCHARMClassifier and type(classifier) != pcalda.PCALDAClassifier:
    print 'Error: unrecognized classifier.'
    sys.exit()

pred_labels = classifier.predict(data_data)

f=open(output_path+'predicted_labels.csv','w')
f.write('Key,Predicted Label\n')
for key, label in zip(data_key, pred_labels):
    f.write(key+','+str(label)+'\n')