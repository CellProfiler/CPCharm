#! /usr/bin/env python
import crossvalidation as cv
import numpy as np
import re
import sys
import csv
import pickle

def saveobject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

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

def format_label(labels):
    formated_labels=np.array(labels)
    #formated_labels=np.array(labels[1:len(labels),:])
    formated_labels[:,[0,1]]=formated_labels[:,[1,0]]
    return formated_labels

data_path=sys.argv[1]
labels_path=sys.argv[2]
output_path=sys.argv[3]
N=int(sys.argv[4])
disp=int(sys.argv[5])
classiftype=sys.argv[6]
cvmethod=sys.argv[7]

if classiftype!='wnd' and classiftype!='lda':
    print 'ERROR: Unrecognized classifier type.'
    sys.exit()
    
if cvmethod!='save25' and cvmethod!='kfold':
    print 'ERROR: Unrecognized cross-validation method.'
    sys.exit()
    
if cvmethod=='kfold':
    k=int(sys.argv[8])
if cvmethod=='save25':
    k=0.25

temp_data=csv.reader(open(data_path, 'rb'), delimiter=',')
temp_labels=csv.reader(open(labels_path, 'rb'), delimiter=',')
lst_data=[]
lst_labels=[]
for row in temp_data: lst_data.append(row)
for row in temp_labels: lst_labels.append(row)
test_data=np.array(lst_data)
test_labels=np.array(lst_labels)

f=open(output_path+'results_summary.txt','w')
f.write("TRAINING/VALIDATION TESTS\n\n")
f.write("SETTINGS:\n")
f.write("Data file: "+data_path+"\n")
f.write("Labels file: "+labels_path+"\n")
f.write("Cross-validation method: "+cvmethod+"\n")
f.write("Classifier type: "+classiftype+"\n")
if cvmethod=='kfold':
    f.write("Parameter k of k-fold cross-validation: "+str(k)+"\n")

test_data=format_data(test_data)
test_labels=format_label(test_labels)

acc=np.array([])
f.write("Training "+str(N)+" classifiers."+"\n"+"\n")

f.write("RESULTS:\n")

if classiftype=='wnd':
    f.write("Top Features:\n")
    
for n in range(0, N):
    if classiftype=='wnd':
        classifier, avgTotal, meaningful_feats = cv.cross_validation(classiftype, cvmethod, test_labels, test_data, k, disp)
    elif classiftype=='lda':
        classifier, avgTotal = cv.cross_validation(classiftype, cvmethod, test_labels, test_data, k, disp)        
    acc=np.append(acc,avgTotal)
    
    if classiftype=='wnd':
        f.write(str(n)+":\n")
        for i in range(0, len(meaningful_feats)):
            f.write(meaningful_feats[i, 0]+"\t"+meaningful_feats[i, 1]+"\n")
        f.write("\n")

if classiftype=='lda':
    saveobject(classifier, output_path+'pcalda_classifier.pk')
if classiftype=='wnd':
    saveobject(classifier, output_path+'wnd_classifier.pk')

f.write("Accuracies:\n")
for n in range(0, N):
    f.write(str(n)+": \t"+str(acc[n])+"\n")

mean=np.mean(acc)
med=np.median(acc)
std=np.std(acc)
f.write("\n")
f.write("Mean performance: "+str(mean)+"\n")
f.write("Median performance: "+str(med)+"\n")
f.write("Standard deviations: "+str(std)+"\n")
