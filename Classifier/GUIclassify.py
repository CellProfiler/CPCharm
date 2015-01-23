#! /usr/bin/env python
import Tkinter
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import os
import inspect
import numpy as np
import re
import sys
import csv
import pickle
import pcalda
import wndcharm

data_file_path = ''
classifier_file_path = ''
output_dir_path = ''

# GUI-related functions
def open_data_file():
  global data_file_path

  previous_data_file_path=data_file_entry.get()
  init_dir=os.path.dirname(previous_data_file_path)
  if not init_dir:
      init_dir=os.path.dirname(classifier_file_entry.get())  

  file_opt = dict(defaultextension ='.csv', filetypes= [('CSV files', '.csv')], initialdir=init_dir)
  filename = askopenfilename(**file_opt)
  
  if filename:
    data_file_path = os.path.normpath(filename)
  else:
    data_file_path=previous_data_file_path
    
  data_file_entry.delete(0, Tkinter.END)
  data_file_entry.insert(0, data_file_path)  

def open_classifier_file():
  global classifier_file_path

  previous_classifier_file_path=classifier_file_entry.get()
  init_dir=os.path.dirname(previous_classifier_file_path)
  if not init_dir:
    init_dir=os.path.dirname(data_file_entry.get())

  file_opt = dict(defaultextension = '.pk', filetypes= [('PK files', '.pk')], initialdir=init_dir)
  filename = askopenfilename(**file_opt)
  
  if filename:
    classifier_file_path = os.path.normpath(filename)
  else:
    classifier_file_path=previous_classifier_file_path
    
  classifier_file_entry.delete(0, Tkinter.END)
  classifier_file_entry.insert(0, classifier_file_path) 

def set_output_dir():
  global output_dir_path

  previous_output_dir_path=output_dir_entry.get()
  init_dir=previous_output_dir_path
  if not init_dir:
      init_dir=os.path.dirname(data_file_entry.get()) 
  if not init_dir:
      init_dir=os.path.dirname(classifier_file_entry.get())       
  
  dir_opt=dict(initialdir=init_dir)
  dirname = askdirectory(**dir_opt)
  
  if dirname: 
    output_dir_path = os.path.normpath(dirname)
  else:
    output_dir_path=previous_output_dir_path
    
  output_dir_entry.delete(0, Tkinter.END)
  output_dir_entry.insert(0, output_dir_path)  

def on_ok():
    global data_path
    global classifier_path
    global output_dir

    data_path=data_file_entry.get()
    classifier_path=classifier_file_entry.get()
    output_dir=output_dir_entry.get()
    root.destroy()

# Input-related functions
def openobject(filename):
  try:
	with open(filename, 'rb') as inputfile:
		obj = pickle.load(inputfile)
	return obj
  except (OSError, IOError) as e:
	print 'ERROR: Invalid classifier.'
	print e
	sys.exit()   

# Formatting-related functions
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
	
# Main
# GUI construction
root = Tkinter.Tk()
root.title("[CP-Charm] Classify Mode")
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

# Classifier entry
classifier_file_label=Tkinter.Label(root,text="Classifier file (.pk):")
classifier_file_label.grid(dict(row=i,column=0,sticky='w'))

classifier_file_entry=Tkinter.Entry(root, width=50, textvariable=classifier_file_path)
classifier_file_entry.grid(dict(row=i,column=1,columnspan=3,sticky='we',padx=4))

classifier_file_button=Tkinter.Button(root, text="Browse", width=10, command=open_classifier_file)
classifier_file_button.grid(dict(row=i,column=4,sticky='e', padx=4))

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

# OK Button
OKbutton = Tkinter.Button(root, width=8)
OKbutton["text"] = "OK"
OKbutton["command"] = on_ok
OKbutton.grid(dict(row=i,column=1, columnspan=2, pady=4))

root.mainloop()

# Actual processing
if 'data_path' in globals() and 'classifier_path' in globals() and 'output_dir' in globals():
  try:
    temp_data=csv.reader(open(data_path, 'rb'), delimiter=',')
  except (OSError, IOError) as e:
    print 'ERROR: Invalid feature file.'
    print e
    sys.exit()   
    
  lst_data=[]
  for row in temp_data: lst_data.append(row)
  data=np.array(lst_data)
  data=format_data(data)

  data_key=data[1:,0]
  data_data=data[1:,1:]

  classifier=openobject(classifier_path)

  if type(classifier) != pcalda.PCALDAClassifier and type(classifier) != wndcharm.WNDCHARMClassifier:
	print 'Error: Unrecognized classifier.'
	sys.exit()
	
  pred_labels = classifier.predict(data_data)

  try:
    f=open(os.path.join(output_dir,'predicted_labels.csv'),'w')
  except (OSError, IOError) as e:
    print 'ERROR: Invalid output directory.'
    print e
    sys.exit()      
 
  f.write('Key,Predicted Label\n')
  for key, label in zip(data_key, pred_labels):
	f.write(key+','+str(label)+'\n')