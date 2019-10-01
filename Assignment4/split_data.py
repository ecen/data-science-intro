import glob
import os
from shutil import copyfile
import ntpath
import re
import random

inputPrefix = "./downloaded_data/"
outputPrefix = "./data/"

spamDirs = ["spam", "spam2"]
easyHamDirs = ["easy_ham", "easy_ham_2"]
hardHamDirs = ["hard_ham"]

def setPrefix(strings):
    return list(map(lambda s : inputPrefix + s, strings))

spamDirs = setPrefix(spamDirs)
easyHamDirs = setPrefix(easyHamDirs)
hardHamDirs = setPrefix(hardHamDirs)

def createSet(dirNames):
    files = []
    for dirName in dirNames:
        files.extend(glob.glob(dirName + "/*"))
    files.sort() # Sort to always have the same ordering
    random.seed(1)
    random.shuffle(files) # Shuffle with a seed
    return files

spamFiles = createSet(spamDirs)
easyHamFiles = createSet(easyHamDirs)
hardHamFiles = createSet(hardHamDirs)

def mkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def emptyDir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def copyFiles(srcPaths, destDir):
    pattern = re.compile("[0-9]+\.*")

    for srcPath in srcPaths:
        basename = ntpath.basename(srcPath)
        if (pattern.match(basename) is None):
            #print(srcPath)
            break
        copyfile(srcPath, destDir + "/" + basename)

def splitSet(setName, fileNames, trainFraction):
    splitIndex = int(trainFraction * len(fileNames))
    train = fileNames[:splitIndex]
    test = fileNames[splitIndex:]

    trainName = outputPrefix + "train_" + setName
    testName = outputPrefix + "test_" + setName
    mkdir(testName)
    mkdir(trainName)
    emptyDir(testName)
    emptyDir(trainName)

    copyFiles(train, trainName)
    copyFiles(test, testName)


splitSet("spam", spamFiles, 0.8)
splitSet("easy_ham", easyHamFiles, 0.8)
splitSet("hard_ham", hardHamFiles, 0.8)
