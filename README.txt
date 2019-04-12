# data preparation 

python data-prep.py

In this step, we extract useful information form steam review raw data.

# Results
All our results that are runnable and reproducible are in two jupyter notebooks

	- naive_bayes.ipynb for naive bayes
	- enhanced_naive_bayes.ipynb for our enhanced naive bayes



** Important notice: I extensively used the f-string feature, which is only available for python >= 3.6.5
   So python >= 3.6.5 is enssential to run our jupyter notebooks.                                        **

Our data file is 1.3Mb for dev, and ~300KB for test, and also, in the second jupyter notebook, we used lexicon data of a english stop words file, which isn't uploaded as well.
Contact us at boweic@usc.edu, bos@usc.edu if you need them, because we don't know should we upload those files to vocareum. With the train/test data, the jupyter notebook should be directly runnable and reproduce all results we had.


Separately, we have the python class definition for the classifier we used in:
	- NB.py for naive bayes
	- ENB.py for our enhanced naive bayes
But since plotting and staff are more convenient in jupyter notebook, so we switch to run our code there.


We also implemented a Byte-Pair Encoding algorithm as described in the stanford text book:
	- byte_pair_encoding.py
But our implementation is too slow when combined with the classifier, so we didn't use it anymore. 
