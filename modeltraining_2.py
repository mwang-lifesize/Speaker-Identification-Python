# python  modeltraining_2.py /var/www/html/record/data speaker_models_lifesize

import sys 
import os 
import cPickle
import numpy as np
from scipy.io.wavfile import read
#from sklearn.mixture import GMM 
from sklearn.mixture import GaussianMixture
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")


# train all dir under this dir_path, and output gmm model to dest_dir
def gmm_train(dir_path, dest_dir):
  for u_dir in os.listdir(dir_path): 
    speaker_name = str(u_dir)
    print "speaker name:" + speaker_name
    user_dir = os.path.join(dir_path, u_dir)
   
    if os.path.isdir(user_dir):
      features = np.asarray(())
      files = os.listdir(user_dir)
      num_files = len(files)
      for file in files:
        file_name = str(file)
        print "\nfile_name : " + file_name
        fname = os.path.join(user_dir, file)
        if os.path.isfile(fname):
          fn = fname.split('/')
          fn = fn[-1]
          # wave file
          if fn[-4:]=='.wav':
            # read the audio
            sr,audio = read(fname)
            # extract 40 dimensional MFCC & delta MFCC features
            vector   = extract_features(audio,sr)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            # when features of 5 files of speaker are concatenated, then do model training
                # -> if count == 5: --> edited below

      # read all sample files , let's gmm
      #gmm = GaussianMixture(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
      gmm = GaussianMixture(n_components = 16, covariance_type='diag')
      gmm.fit(features)
      # dumping the trained gaussian model
      picklefile = speaker_name+".gmm"
      cPickle.dump(gmm,open(dest_dir + '/' + picklefile,'w'))
      print '+ modeling completed for speaker:',picklefile," with data point = ",features.shape    


source_dir = sys.argv[1]
dest_dir = sys.argv[2]
print(" training dir:", source_dir)
print(" output gmm to :", dest_dir)

gmm_train( source_dir, dest_dir )
            
