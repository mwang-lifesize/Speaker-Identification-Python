import os
import sys
import cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

"""
#path to training data
source   = "development_set/"   
modelpath = "speaker_models/"
test_file = "development_set_test.txt"        
file_paths = open(test_file,'r')

"""
#path to training data
source   = "SampleData/"   

#path where training speakers will be saved
#modelpath = "Speakers_models/"
modelpath = "speaker_models_lifesize/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

error = 0
total_sample = 0.0


#path = raw_input().strip()   
path =  sys.argv[1]
   
print "Testing Audio : ", path
#sr,audio = read(source + path)
sr,audio = read(path)
vector   = extract_features(audio,sr)

log_likelihood = np.zeros(len(models)) 

for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        print( i,  ", scores:", scores)
        log_likelihood[i] = scores.sum()

winner = np.argmax(log_likelihood)

print( " log_like is:", log_likelihood[winner])
print "\tdetected as - ", speakers[winner]



print "Hurrey ! Speaker identified. Mission Accomplished Successfully. "
