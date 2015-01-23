#! /usr/bin/env python
import crossvalidation as cv
import numpy as np
import re
import sys
import csv
import pickle
import Tkinter
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import os
import inspect

data_file_path = ''
labels_file_path = ''
output_dir_path = ''

# GUI-related functions
def open_data_file():
  global data_file_path

  previous_data_file_path=data_file_entry.get()
  init_dir=os.path.dirname(previous_data_file_path)
  if not init_dir:
    init_dir=os.path.dirname(labels_file_entry.get())
  
  file_opt = dict(defaultextension ='.csv', filetypes= [('CSV files', '.csv')], initialdir=init_dir)
  filename = askopenfilename(**file_opt)
  
  if filename:
    data_file_path = os.path.normpath(filename)
  else:
    data_file_path=previous_data_file_path
    
  data_file_entry.delete(0, Tkinter.END)
  data_file_entry.insert(0, data_file_path)   

def open_labels_file():
  global labels_file_path

  previous_labels_file_path=labels_file_entry.get()
  init_dir=os.path.dirname(previous_labels_file_path)
  if not init_dir:
    init_dir=os.path.dirname(data_file_entry.get())

  file_opt = dict(defaultextension = '.csv', filetypes= [('CSV files', '.csv')], initialdir=init_dir)
  filename = askopenfilename(**file_opt)
  
  if filename:
    labels_file_path = os.path.normpath(filename)
  else:
    labels_file_path=previous_labels_file_path
    
  labels_file_entry.delete(0, Tkinter.END)
  labels_file_entry.insert(0, labels_file_path)  

def set_output_dir():
  global output_dir_path

  previous_output_dir_path=output_dir_entry.get()
  init_dir=previous_output_dir_path
  if not init_dir:
      init_dir=os.path.dirname(data_file_entry.get()) 
  if not init_dir:
      init_dir=os.path.dirname(labels_file_entry.get())       
  
  dir_opt=dict(initialdir=init_dir)
  dirname = askdirectory(**dir_opt)
  
  if dirname: 
    output_dir_path = os.path.normpath(dirname)
  else:
    output_dir_path=previous_output_dir_path
    
  output_dir_entry.delete(0, Tkinter.END)
  output_dir_entry.insert(0, output_dir_path)  

def enable_allfolds():
  folds_button1.configure(state='normal')
  folds_button2.configure(state='normal')
  if folds_button_value.get()=='k':
    folds_spin.configure(state='normal')

def disable_allfolds():
  folds_button1.configure(state='disabled')
  folds_button2.configure(state='disabled')
  folds_spin.configure(state='disabled')

def enable_foldspin():
  folds_spin.configure(state='normal')

def disable_foldspin():
    folds_spin.configure(state='disabled')
    
def on_ok():
    global data_path
    global labels_path
    global output_dir
    global classifier_type
    global validation_type
    global holdout
    global display
    global repeats
    global folds
    global folds_type

    data_path=data_file_entry.get()
    labels_path=labels_file_entry.get()
    output_dir=output_dir_entry.get()
    classifier_type=classifier_type_value.get()
    validation_type=validation_type_value.get()
    folds_type=folds_button_value.get()
    folds=folds_spin.get()
    display=display_type_value.get();
    holdout=holdout_type_value.get();
    repeats=repeats_spin.get()
    root.destroy()

# Output-related functions
def saveobject(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Formatting-related functions
def format_data(data):
	formated_data=np.array(data)
	
	ind_key=-1
	for i in range(0, len(formated_data[0,:])):
		if re.search("[Kk]ey", formated_data[0,i]):
			ind_key=i

	if ind_key<0:
		print 'ERROR: Image identifiers are absent from the features file.'
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
		print 'ERROR: Image identifiers are absent from the labels file.'
		sys.exit()
		
	if ind_class<0:
		print 'ERROR: Class identifiers are absent from the labels file.'
		sys.exit()
	if holdout:
		if ind_holdout<0:
			print 'ERROR: HoldOut mode selected, but hold out metadata are absent from the labels file.'
			sys.exit()
				
	if holdout:
		formated_labels[:,[0,1,2]]=formated_labels[:,[ind_class,ind_holdout,ind_key]]
	else:
		formated_labels[:,[0,1]]=formated_labels[:,[ind_class,ind_key]]
			
	return formated_labels

# Main
# GUI construction
root = Tkinter.Tk()
root.title("[CP-Charm] TrainTest Mode")
Tkinter.Grid.columnconfigure(root,3,weight=1)
i=0

# Data entry
data_file_label=Tkinter.Label(root,text="Features file (.csv):")
data_file_label.grid(dict(row=i,column=0,sticky='w'))

data_file_entry=Tkinter.Entry(root, width=50, textvariable=data_file_path)
data_file_entry.grid(dict(row=i,column=1,columnspan=3,sticky='we',padx=4))

data_file_button=Tkinter.Button(root, text="Browse", width=10, command=open_data_file)
data_file_button.grid(dict(row=i,column=4,sticky='e', padx=4))

i=i+1

# Labels entry
labels_file_label=Tkinter.Label(root,text="Labels file (.csv):")
labels_file_label.grid(dict(row=i,column=0,sticky='w'))

labels_file_entry=Tkinter.Entry(root, width=50, textvariable=labels_file_path)
labels_file_entry.grid(dict(row=i,column=1,columnspan=3,sticky='we',padx=4))

labels_file_button=Tkinter.Button(root, text="Browse", width=10, command=open_labels_file)
labels_file_button.grid(dict(row=i,column=4,sticky='e', padx=4))

i=i+1

# Output
output_dir_label=Tkinter.Label(root,text="Output directory:")
output_dir_label.grid(dict(row=i,column=0,sticky='w'))

output_dir_entry=Tkinter.Entry(root, width=50, textvariable=output_dir_path)
output_dir_entry.insert(0,os.path.dirname(inspect.getsourcefile(open_data_file)))
output_dir_entry.grid(dict(row=i,column=1,columnspan=3,sticky='we',padx=4))

output_dir_button=Tkinter.Button(root, text="Browse", width=10, command=set_output_dir)
output_dir_button.grid(dict(row=i,column=4,sticky='e', padx=4))

i=i+1

# Classifier type
classifier_type_label=Tkinter.Label(root,text="Classifier type:")
classifier_type_label.grid(dict(row=i,column=0,sticky='w'))

CLASSIFIERS = [
      ("WND", "wnd"),
      ("PCA-LDA", "lda"),
]

classifier_type_value = Tkinter.StringVar()
classifier_type_value.set('lda')

j=1
for text, mode in CLASSIFIERS:
      classifier_type_button = Tkinter.Radiobutton(root, text=text, variable=classifier_type_value, value=mode)
      classifier_type_button.grid(dict(row=i,column=j,sticky='w'))
      j=j+1        

i=i+1

# Validation type
validation_type_label=Tkinter.Label(root,text="Validation type:")
validation_type_label.grid(dict(row=i,column=0,sticky='w'))
      
validation_type_value = Tkinter.StringVar()
validation_type_value.set('kfold')

validation_type_button = Tkinter.Radiobutton(root, text="Leave 25% out", variable=validation_type_value, value="save25", command=disable_allfolds)
validation_type_button.grid(dict(row=i,column=1,sticky='w'))
validation_type_button = Tkinter.Radiobutton(root, text="k-fold cross-validation", variable=validation_type_value, value="kfold", command=enable_allfolds)
validation_type_button.grid(dict(row=i,column=2,sticky='w'))

i=i+1

# Folds
folds_label=Tkinter.Label(root,text="Cross-validation folds type:")
folds_label.grid(dict(row=i,column=0,sticky='w'))

folds_button_value = Tkinter.StringVar()
folds_button_value.set('k')

folds_button1 = Tkinter.Radiobutton(root, text="All-against-one", variable=folds_button_value, value="allvsone", command=disable_foldspin)
folds_button1.grid(dict(row=i,column=1,sticky='w'))
folds_button2 = Tkinter.Radiobutton(root, text="k =", variable=folds_button_value, value="k", command=enable_foldspin)
folds_button2.grid(dict(row=i,column=2,sticky='w'))

folds_default=Tkinter.StringVar()
folds_default.set("10")
folds_spin=Tkinter.Spinbox(root, from_=1, to=5000, textvariable=folds_default)
folds_spin.grid(dict(row=i,column=3,sticky='w'))

i=i+1

# Display
display_type_label=Tkinter.Label(root,text="Display confusion matrices:")
display_type_label.grid(dict(row=i,column=0,sticky='w'))
    
DISPLAYOPTIONS = [
  ("Yes", "1"),
  ("No", "0"),
]
      
display_type_value = Tkinter.StringVar()
display_type_value.set('1')

j=1
for text, mode in DISPLAYOPTIONS:
  display_type_button = Tkinter.Radiobutton(root, text=text, variable=display_type_value, value=mode)
  display_type_button.grid(dict(row=i,column=j,sticky='w'))
  j=j+1  

i=i+1

# Holdout      
holdout_type_label=Tkinter.Label(root,text="Include hold-out samples information:")
holdout_type_label.grid(dict(row=i,column=0,sticky='w'))
    
HOLDOUTOPTIONS = [
  ("Yes", "1"),
  ("No", "0"),
]
      
holdout_type_value = Tkinter.StringVar()
holdout_type_value.set('1')

j=1
for text, mode in HOLDOUTOPTIONS:
  holdout_type_button = Tkinter.Radiobutton(root, text=text, variable=holdout_type_value, value=mode)
  holdout_type_button.grid(dict(row=i,column=j,sticky='w'))
  j=j+1

i=i+1

# Repeats
repeats_label=Tkinter.Label(root,text="Number of training/testing rounds:")
repeats_label.grid(dict(row=i,column=0,sticky='w'))

repeats_spin=Tkinter.Spinbox(root, from_=1, to=1000)
repeats_spin.grid(dict(row=i,column=1,sticky='w'))

i=i+1

# OK Button
OKbutton = Tkinter.Button(root, width=8)
OKbutton["text"] = "OK"
OKbutton["command"] = on_ok
OKbutton.grid(dict(row=i,column=1, columnspan=2, pady=4))

root.mainloop()

# Actual processing
if 'data_path' in globals() and 'labels_path' in globals() and 'output_dir' in globals() and 'classifier_type' in globals() and 'validation_type' in globals() and 'holdout' in globals() and 'display' in globals() and 'repeats' in globals() and 'folds' in globals() and 'folds_type' in globals():
  N=int(repeats)
  holdout=int(holdout)
  disp=int(display)
  
  if classifier_type!='wnd' and classifier_type!='lda':
	  print 'ERROR: Unrecognized classifier type.'
	  sys.exit()
      
  if validation_type!='save25' and validation_type!='kfold':
	  print 'ERROR: Unrecognized cross-validation method.'
	  sys.exit()
	  
  if(validation_type=='kfold'):
    if(folds_type=='k'):
      k=int(folds)
    else:
      k=None
  else:
    k=0.25
  
  try:
    temp_data=csv.reader(open(data_path, 'rb'), delimiter=',')
  except (OSError, IOError) as e:
    print 'ERROR: Invalid feature file.'
    print e
    sys.exit()
    
  try:
    temp_labels=csv.reader(open(labels_path, 'rb'), delimiter=',')
  except (OSError, IOError) as e:
    print 'ERROR: Invalid labels file.'
    print e
    sys.exit()    
    
  lst_data=[]
  lst_labels=[]
  for row in temp_data: lst_data.append(row)
  for row in temp_labels: lst_labels.append(row)
  test_data=np.array(lst_data)
  test_labels=np.array(lst_labels)
  
  try:
    f=open(os.path.join(output_dir,'results_summary.txt'),'w')
  except (OSError, IOError) as e:
    print 'ERROR: Invalid output directory.'
    print e
    sys.exit()        

  f.write("TRAINING/VALIDATION TESTS\n\n")
  f.write("SETTINGS:\n")
  f.write("Data file: "+data_path+"\n")
  f.write("Labels file: "+labels_path+"\n")
  f.write("Cross-validation method: "+validation_type+"\n")
  f.write("Classifier type: "+classifier_type+"\n")
  if validation_type=='kfold':
	  f.write("Parameter k of k-fold cross-validation: "+str(k)+"\n")
  
  test_data=format_data(test_data)
  test_labels=format_label(test_labels, holdout)
  
  acc=np.array([])
  f.write("Training "+str(N)+" classifiers."+"\n"+"\n")
  
  f.write("RESULTS:\n")
  
  for n in range(0, N):
	  classifier, avgTotal = cv.cross_validation(classifier_type, validation_type, test_labels, test_data, k, disp, holdout)		
	  acc=np.append(acc,avgTotal)
  
  if classifier_type=='lda':
	  saveobject(classifier, os.path.join(output_dir,'pcalda_classifier.pk'))
  if classifier_type=='wnd':
	  saveobject(classifier, os.path.join(output_dir,'wnd_classifier.pk'))
  
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
