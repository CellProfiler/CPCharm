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
	formated_data=np.array(data)
	
	ind_key=-1
	for i in range(0, len(formated_data[0,:])):
		if re.search("[Kk]ey", formated_data[0,i]):
			ind_key=i

	if ind_key<0:
		print 'ERROR: Image identifiers are absent.'
		sys.exit()

	formated_data[:,[0,ind_key]]=formated_data[:,[ind_key,0]]	 	
	return np.copy(formated_data)

def format_label(labels, holdout):
	formated_labels=np.array(labels)
	if holdout:
		if(len(formated_labels[0,:])>3):
			print 'ERROR: Label file should contain 3 columns (Key, Class and HoldOut). '+str(len(formated_labels[0,:]))+' were detected.'
			sys.exit()
	else:
		if(len(formated_labels[0,:])>2):
			print 'ERROR: Label file should contain 2 columns (Key, Class). '+str(len(formated_labels[0,:]))+' were detected.'
			sys.exit()
				
	#formated_labels=np.array(labels[1:len(labels),:])
	
	ind_key=-1
	ind_class=-1
	ind_holdout=-1
	for i in range(0, len(formated_labels[0,:])):
		if re.search("[Kk]ey", formated_labels[0,i]):
			ind_key=i
		if re.search("[Cc]lass", formated_labels[0,i]):
			ind_class=i
		if re.search("[Hh]old[Oo]ut", formated_labels[0,i]):
			ind_holdout=i

	if ind_key<0:
		print 'ERROR: Image identifiers are absent.'
		sys.exit()
		
	if ind_class<0:
		print 'ERROR: Class identifiers are absent from label file.'
		sys.exit()
	if holdout:
		if ind_holdout<0:
			print 'ERROR: HoldOut mode selected, but hold out metadata absent from label file.'
			sys.exit()
				
	if holdout:
		formated_labels[:,[0,1,2]]=formated_labels[:,[ind_class,ind_holdout,ind_key]]
	else:
		formated_labels[:,[0,1]]=formated_labels[:,[ind_class,ind_key]]
			
	# OLD formated_labels[:,[0,1]]=formated_labels[:,[1,0]]
	return formated_labels

data_path=sys.argv[1]
labels_path=sys.argv[2]
output_path=sys.argv[3]
N=int(sys.argv[4])
holdout=int(sys.argv[5])
disp=int(sys.argv[6])
classiftype=sys.argv[7]
cvmethod=sys.argv[8]

if classiftype!='wnd' and classiftype!='lda':
	print 'ERROR: Unrecognized classifier type.'
	sys.exit()
    
if cvmethod!='save25' and cvmethod!='kfold':
	print 'ERROR: Unrecognized cross-validation method.'
	sys.exit()
    
if cvmethod=='kfold':
	k=int(sys.argv[9])
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
test_labels=format_label(test_labels, holdout)

acc=np.array([])
f.write("Training "+str(N)+" classifiers."+"\n"+"\n")

f.write("RESULTS:\n")

for n in range(0, N):
	classifier, avgTotal = cv.cross_validation(classiftype, cvmethod, test_labels, test_data, k, disp, holdout)		
	acc=np.append(acc,avgTotal)

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
