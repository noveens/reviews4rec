Compile using "make".

You will need to export liblbfgs to your LD_LIBRARY_PATH to run the code (export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/liblbfgs-1.10/lib/.libs/).

Run using ./train and specifying an input file (e.g. the Artz.votes.gz file provided). Input files should be a list of quadruples of the form (userID, itemID, rating, time) followed by the number of words for that review, followed by the words themselves (see Arts.votes.gz).

For additional datasets see snap.stanford.edu/data

Questions and comments to julian.mcauley@gmail.com
