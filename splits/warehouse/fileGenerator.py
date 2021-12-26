import glob

# put the path of scratch folder
mypath = "/home2/pranjalipathre/RackLay/data/Data/*"
train_file_name = "train_temporal_files.txt"
test_file_name = "val_temporal_files.txt"

# Get the list of all sequences in 
seqLists = glob.glob(mypath)

# How many sequences are present?
lenSeqLists = len(seqLists)
# make train file
train_file = open(train_file_name, 'w+')
for i in range(len(seqLists) - 1):
    imgPaths = glob.glob(seqLists[i] + "/img/*")
    for path in imgPaths:
        train_file.write(path + "\n")
    train_file.write(",")

train_file.close()

# HARD CODE PART
# make last seq as test seq
test_file = open(test_file_name, 'w+')
imgPaths = glob.glob(seqLists[len(seqLists)-1] + "/img/*")
for path in imgPaths:
    test_file.write(path + "\n")
test_file.write(",")
test_file.close()
