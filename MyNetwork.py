import numpy as np
import sys
import time
def writetofile(layer1, layer2):
	w = open("netWeights.txt","w")
	w.write("LAYER 1\n")
	np.savetxt(w,layer1)
	

	w.write('\n')
	w.write("Layer 2\n")
	np.savetxt(w,layer2)
	
def readnetW(filename,testfile,testlabels):
	print "Reading netWeights"
	readfileobj = open(filename,'r')
	line = readfileobj.readline()
	layer1=[]

	if line=="LAYER 1\n":
	   print "Layer 1 found"
	   line = readfileobj.readline()   
	   while line!="Layer 2\n":
	   	line = line.split()
	   	line = map(float,line)
	   	if (line!=[]):
	   	   layer1.append(line)
	   	line= readfileobj.readline()
	layer1 = np.array(layer1)


	layer2 = []
	if line=="Layer 2\n":
	   print "Layer 2 found"
	   while line !=[]:
		   line = readfileobj.readline()
		   line= line.split()
		   line = map(float,line)
		   if (line!=[]):
		   	 layer2.append(line)

	layer2 = np.array(layer2)
	testfunction(testfile,testlabels,layer1,layer2)	
		
	

		   	






def err_i2h(diff,wji,hiddenlayer,inp):
	
	inp = np.array([inp]).transpose()
	
	difference = (np.array([diff])).transpose()
	
	
	expression1 = hiddenlayer *(1-hiddenlayer) # hj*(1-hj)
	expression1 = np.array([expression1]) # 1 x 30 
	

	# wji = wji.transpose()
	expression2 = np.dot(wji,difference) # weights of forward layer into diff
	

	expression3 = np.multiply(expression1,expression2.transpose())
	
	returnexp= np.dot(inp,expression3)
	
	return returnexp


def err_h2o(diff,hiddenlayer):
	
	h_layer =  np.array([hiddenlayer]) # 1 x 30
	
	h_layer = h_layer.transpose() # 30 x 1 
	
	diff =  np.array([diff]) # 1 x 10 
	
	err = np.matmul(h_layer,diff)
	
	# print err
	return err	

def retlabel(layer):
	maxno = layer[0]
	maxindex = 0
	for x in xrange(0,len(layer)):
		if (layer[x]>maxno):
			maxno = layer[x]
			maxindex = x

	return maxindex		
			
		

def testfunction(testfile ,label,layer1,layer2):
	testdata = fileread(testfile)
	testlabels = readlabels(label)
	
	testdata_np = np.array(testdata)
	testdata_np= testdata_np/float(255)
	testlabels_np = np.array(testlabels)

	
	hits = 0
	for i in xrange(0,len(testlabels_np)):
		middle =  np.dot(testdata_np[i],layer1)
		middle = sigmoid(middle)
		outputlayer = np.dot(middle,layer2)
		outputlayer = sigmoid(outputlayer)
		label = retlabel(outputlayer)
		if (testlabels_np[i] == label):
			hits = hits + 1 
		


	print "TEST COMPLETE \nAccuracy report:"		
	print "Hits = " ,hits,'out of', len(testlabels_np)
	perc = hits*100/len(testlabels_np)
	print "Accuracy perecentage is = ", perc 
	print "Error perecentage is = ", (100-perc) 
	
	
		
def sigmoid(wx):
	# wx = np.clip( wx, -500, 500 )
	exp = (1.000+np.exp(-wx))
	ret = 1.0000/exp
	ret1 = np.array(ret, dtype = float)
	return ret1 

	 
def targetvec(digit):
	nlist = []
	for x in xrange(0,10):
		nlist.append(0)
	nlist[digit] = 1
	nlist = np.array(nlist)
	return nlist		

def errorfunction(size,targetdigit,act_layer):
	targetvector = targetvec(targetdigit)
	temp = 1 - targetvector 
	temp2 = 1 - act_layer
	error = targetvector.dot(np.log(act_layer)) + temp.dot(np.log(temp2))
	errorret =  error/(-size)
	return errorret


def readhelper1(check):
	ret =""
	for x in xrange(0,len(check)):
		if (check[x] == ']'): return ret
		else:
			ret = ret + check[x] 


	return False	

def fileread(name):
	readfileobj = open(name, 'r')
	line = readfileobj.readline()
	newlist = []
	mainrec = []
	print "Reading training data"
	while line != '' :
		while readhelper1(line)==False :
			line= line.split()	
			if (line[0]=='['): 
				# print "NEW RECORD"
				for x in xrange(1,len(line)):
					newlist.append(int(line[x]))
				line =readfileobj.readline()	
					
			else:		
				
				for x in xrange(0,len(line)):
					newlist.append(int(line[x]))	
				line = readfileobj.readline()

				if readhelper1(line)!=False:
					end = readhelper1(line)				
					line = end.split()
					
					for x in xrange(0,len(line)):
						newlist.append(int(line[x]))
					# print "RECORD complete"	
					break
		# print "RECORD appended"
		mainrec.append(newlist)	
		newlist = []			
		line = readfileobj.readline()			


	# print len(mainrec)
	return mainrec
	
def readlabels(name):
	readfileobj = open(name, 'r')
	print "Reading labels"
	line = readfileobj.readline()
	labels = []
	while line != '':
		labels.append(int(line))
		line = readfileobj.readline()
	return labels	

def main(trainfile,labelfile,learnR):
	traindata = fileread(trainfile)
	trainlabels = readlabels(labelfile)
	
	traindata_np = np.array(traindata)
	traindata_np= traindata_np/float(255)
	

	trainlabels_np = np.array(trainlabels)
	
	# print len(traindata)
	# print len(trainlabels)

	
	print "Initializing weights"
	layer1 = 2*np.random.random((784,30)) - 1 
	layer2 = 2*np.random.random((30,10)) - 1
	l1 = layer1
	l2 =layer2
	
	
		
	print "Start training"
	# for len of traindata_np
	for e in xrange(0,2):
		
		for i in xrange(0,len(trainlabels)):
			
			hiddenlayer = np.dot(traindata_np[i],layer1)
			hiddenlayer = sigmoid(hiddenlayer)
			

			outputlayer =  np.dot(hiddenlayer,layer2) #dot
			
			activation =  sigmoid(outputlayer)
			

			diff = activation - targetvec(trainlabels_np[i])
			totalerror = errorfunction(len(trainlabels),trainlabels_np[i],activation)
			if (totalerror<0.0000001):
				
				break	
			# Hidden
			
			error_h= err_h2o(diff,hiddenlayer)
			deltalayer2 = learnR*error_h
			layer2 = layer2 - deltalayer2   # update hidden layer [30x10]
			
			# input to hidden error compute
			
			error_I = err_i2h(diff,layer2,hiddenlayer,traindata_np[i])
			deltalayer1 = learnR*error_I
			layer1 = layer1 - deltalayer1
		
		print "EPOCH "
		print e+1
				
	writetofile(layer1,layer2)		
	print "Training End"

	# testfunction('test.txt','test-labels.txt',layer1,layer2)	
	

		





	return




				
# main('newrec.txt', 'newlabels.txt',0.01)

mode = sys.argv[1]
if mode=='train':
	trainfile = sys.argv[2]
	trainlabelsfile = sys.argv[3]
	learnR = float(sys.argv[4])
	starttime = time.time()
	main(trainfile, trainlabelsfile,learnR)
	print "Training Time is = ", (time.time()-starttime)
	 
else:
	if mode=='test':
		print"TEST"
		testfile = sys.argv[2]
		testlabels= sys.argv[3]
		netWeightsFile = sys.argv[4]
		readnetW(netWeightsFile,testfile,testlabels)