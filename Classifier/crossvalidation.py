#!/usr/bin/env python

'''
Cross validation.

'''

import sys
import numpy as np
import pcalda

def _k_fold_cross_validation_iterator(data, K=10):
	"""
	Generates K (training, validation) pairs from the items in X. If K is unspecified it is set to a leave one out
	"""
	if K == None:
		K = data.shape[0]

	np.random.shuffle(data)
	for k in xrange(K):
		training = np.array([x for i, x in enumerate(data) if i % K != k], dtype='S100')
		validation = np.array([x for i, x in enumerate(data) if i % K == k], dtype='S100')
		yield training, validation

def get_confusion_matrix(ckdata, headers, holdout=False):
	'''	
	return classes, confusion, percent, avg, avgTotal
	
	compute a confusion matrix from the 'K'-fold validation of the 'classifier' on the 'cdata', cdata contains class labels in the first column
	'''
	
	classes = np.sort(np.unique(ckdata[:,0]))
	numclasses = classes.shape[0]
	confusion = np.zeros((numclasses,numclasses), dtype=np.int)
	
	cross_validation_iterator=_k_fold_cross_validation_iterator

	for training, test in cross_validation_iterator(ckdata):
		#print 'Processing...'
		training_labels = training[:,0]
		if holdout:
			training_type=training[:,1]
			#training_keys = training[:,2]
			training_data	= training[:,3:]
		else:
			#training_keys = training[:,1]
			training_data	= training[:,2:]
	
		test_labels = test[:,0]
		if holdout:
			test_type = test[:,1]
			#test_keys = test[:,2]
			test_data	= test[:,3:]
		else:
			#test_keys = test[:,1]
			test_data	= test[:,2:]

		classifier=pcalda.PCALDAClassifier()
		if holdout:
			pred_labels=np.zeros([len(test_labels)],dtype=test_labels.dtype)
			types=np.unique(test_type)

			for i in range(0,len(types)):
				test_ind=np.array((np.where(test_type==types[i]))[0])
				test_data_subset=test_data[test_ind]
				test_labels_subset=test_labels[test_ind]
					
				training_ind=np.where(training_type!=types[i])
				training_labels_subset=training_labels[training_ind]
				training_data_subset=training_data[training_ind]
					
				for j in range(0,len(test_labels_subset)):
					if test_labels_subset[j] not in training_labels_subset:
						print 'WARNING: the class of some elements you will try to validate is not represented on the training set. You might want to check your hold-out metadata.' 
				
				
				classifier.train(training_labels_subset, training_data_subset)
				pred_labels[test_ind] = classifier.predict(test_data_subset)
		else:
			#print 'Training...'
			classifier.train(training_labels, training_data)	
			#print 'Training done.' 
			
			#print 'Testing...'
			pred_labels = classifier.predict(test_data)
			#print 'Testing done.'			
		
		for i, label_pred in enumerate(pred_labels):
			label_real = test_labels[i]
			ind_real = np.where(classes==label_real)
			ind_pred = np.where(classes==label_pred)
			confusion[ind_real, ind_pred] += 1
	
	s = np.sum(confusion,axis=1)
	percent = [100*confusion[i,i]/float(s[i]) for i in range(len(s))]
	#avg = np.mean(percent)
	avgTotal = 100 * np.trace(confusion) / float(np.sum(confusion))
	
	return classifier, classes, confusion, percent, avgTotal

def _inner_join(labels, profiles, keysize):
	for labelrow in labels:
		for profilerow in profiles:
			k1 = labelrow[(1+keysize)]
			k2 = profilerow[0]
			#print str(k1) + ' => ' + str(k2)
			if (k1==k2):
				row = list(labelrow[:(1+keysize)]) + list(profilerow[:])
				yield row

def cross_validation(labels_csvfile, profile_csvfile, displayresult=True, holdout=False):
	'''
	- labels_csvfile's first column are the labels, following N columns are the key 
	- profile_csvfile N first column are considered to be the key, the remining column are the vector data values
	- First row (headers) in two files are ignored
	'''
		
	#labels	= [row for row in csv.reader(open(labels_csvfile, "rb"))]
	#profiles = [row for row in csv.reader(open(profile_csvfile, "rb"))]
	
	labels = np.array(labels_csvfile)
	profiles = np.array(profile_csvfile)
	#print 'labels: ', labels.shape
	#print 'profiles:', profiles.shape
	
	# ignore header row
	headers = profiles[0,:]
	labels = labels[1:,:]
	profiles = profiles[1:,:]

	#keysize = labels.shape[1] - 1
	headers = headers[1:]
	ckdata = np.array([row for row in _inner_join(labels, profiles, holdout)])
	
	# remove nan rows
	nan_row_indices = np.unique(np.where(ckdata == 'nan')[0])
	ckdata = np.delete(ckdata, nan_row_indices, axis=0)

	classifier, classes, confusion, percent, avgTotal = get_confusion_matrix(ckdata, headers, holdout)		

	if displayresult:
		_display_as_text(classes, confusion, percent, avgTotal, ckdata.shape[0])
		#_display_as_graph(classes, confusion, percent, avgTotal)
	
	return classifier, avgTotal

def concentration_selection(compound1, compound2, compoundKeyIndex):
	return None
	
def _display_as_text(classes, confusion, percent, avg, numprofiles):
	
	s = np.sum(confusion,axis=1)
	print 'Classifying %d profiles:' % numprofiles
	for i, v in enumerate(confusion):
		for u in v:
			print "%2d " % u ,
		print "(%02d)  %3d%%  %s " % (s[i],percent[i],classes[i])
	
	print 'Average: %3d%%' % avg

#def _display_as_graph(classes, confusion, percent, avg, filename=None):	
	#from mpl_toolkits.axes_grid.inset_locator import inset_axes
	
	## normalized figure
	#s = np.sum(confusion,axis=1)
	#norm_confusion = np.zeros((len(classes),len(classes)), dtype=np.float)
	#for i, row in enumerate(confusion):
		#for j, v in enumerate(row):
			#norm_confusion[i,j] = v / float(s[i])
		
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#im = ax.imshow(norm_confusion, interpolation='nearest', cmap = mpl.cm.RdBu_r)

	#axins = inset_axes(ax, width="5%", height="50%", loc=3, bbox_to_anchor=(0., 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)  
	#plt.colorbar(im, cax=axins, ticks=[1, max(s)])

	#ax.set_xticklabels([''])
	#ax.set_yticklabels([''])
	
	## print categories
	#xcoord = -1 #len(classes)
	#ycoord = 0
	#ticks = []
	#for i in range(len(classes)):
		##ticks.append(yoffset)
		#txt = "(%02d)  %3d%%  %s " % (s[i],percent[i],classes[i])
		#ax.text(xcoord, ycoord, txt, transform=ax.transData, fontsize='small', verticalalignment='center', horizontalalignment='right')
		#ycoord += 1
	
	#plt.xticks([], [])
	#plt.yticks(ticks, [])
	
	##plt.draw()
	#if filename != None:
		#plt.savefig("%s.png"%filename, format='png')
	#else:
		#plt.show()
