# Assembling feature vectors for the second try on GraphSAGE.
# Follow this procedure:
# Read in files: ID's and subreddit files and concepts
# Then prep args, and then read in the subreddit dicts for each slice
# Do a quick check to make sure all is the same between NumPy arrays and dicts
# Make 3 empty stores for new feature vectors in Numpy
# Loop over concepts
# For each concept, open the numpy array for the files.
# Do the necessary normalisations/logs etc and add to the main numpy array
# Repeat for all 6 concepts, so each of the 3 slices has now a normalised, log numpy matrix. Return those
# Then develop this by considering the difference features.
# For each pair of slices, find the intersection of those subreddits.
# Then make new numpy arrays for those, take existing values over as they should be.
# Then add the difference elements. Return those too (x2)
# Then finally for intersection of all 3

# Left: check gzip opening, labels


# 1. Then feature vector store should be a new matrix.
# 2. First establish features just for the slice: 4,5,6 set, for example.
# 3. Find the intersection of subs between each pair (to start with)
# 4. Loop over the subs and for each, give a 12 element feature
# 5. First 6 elements are for each concept, where the value is the log of the ratio of concept mentions to total token mentions.
# 6. In the case that there were 0 mentions, we obviously can't take the log of 0. So artificially increase by 1 (the count) and then normalise.
# 7. Second 6 elements are for the difference between that concept on that slice and the previous slice. (difference of logs).
# 8. Generating labels here too.

# Read in files: ID's and subreddit files and concepts

### !!! - Intersting issue: add one smoothing could flip a real situation into something weird. !!!

import numpy as np
import sys
import gzip
import json
import math

def normaliser(list_o_subs, origMatrix, totalMatrx, newMatrix, cIdx, rawMatrix):
    for i in range(0,len(list_o_subs)):
        # For each subreddit, set the value of the featsBegin to be the value of the begin_mat divided by the baseArray
        beginVal = np.sum(origMatrix[i,:14])
        if beginVal == 0:
            beginVal += 1
        beginTotVal = np.sum(totalMatrx[i,:14])

        newMatrix[i][cIdx] = math.log(beginVal/beginTotVal)
        newMatrix[i][6] = origMatrix[i,14]

        rawMatrix[i][cIdx] = beginVal/beginTotVal
        rawMatrix[i][6] = origMatrix[i,14]


    return newMatrix, rawMatrix

def normer(secondMat, firstMat, cIndY, second_map, first_map, subName, pUpped):
    # return the specific label
    init = firstMat[first_map[subName]][cIndY]
    diff = secondMat[second_map[subName]][cIndY] - init
    ratiod = diff/init

    if (ratiod * 100) > pUpped:
        return 1
    else:
        return 0



script_name, beginning, middle, end, concepts_file, percentUp = sys.argv

# Then prep args, and then read in the subreddit dicts for each slice

if not beginning.startswith('begin=') or not middle.startswith('mid=') or not end.startswith('end='):
    print("Wrong file format for entry. Exit...", flush=True)
    sys.exit()
if not concepts_file.startswith('concepts'):
    print("Wrong format.", flush=True)
    sys.exit()
if not percentUp.startswith('pIncrease'):
    print("Wrong format.", flush=True)
    sys.exit()

concept_list = []

beginARG = beginning.split('=')[1]
middleARG = middle.split('=')[1]
endARG = end.split('=')[1]

beginID = beginARG.split('!')[0]
beginSubsName = beginARG.split('!')[1]
middleID = middleARG.split('!')[0]
middleSubsName = middleARG.split('!')[1]
endID = endARG.split('!')[0]
endSubsName = endARG.split('!')[1]

pUp = int(percentUp.split('=')[1])


with gzip.open(beginSubsName,'rb') as begPtr:
    for line in begPtr:
        subStr = line.decode('utf-8')
        beginDict = json.loads(subStr)
        beginDictSubs = beginDict['alphabetised_subreddits']
        break

with gzip.open(middleSubsName, 'rb') as midPtr:
    for line in midPtr:
        sub2Str = line.decode('utf-8')
        midDict = json.loads(sub2Str)
        midDictSubs = midDict['alphabetised_subreddits']
        break

with gzip.open(endSubsName, 'rb') as endPtr:
    for line in endPtr:
        sub3Str = line.decode('utf-8')
        endDict = json.loads(sub3Str)
        endDictSubs = endDict['alphabetised_subreddits']
        break

# Do a quick check to make sure all is the same between NumPy arrays and dicts - read in base case first and check with that. Will repeat checks each time too.
baseBegin = np.load('/scratch/orie3748/GraphSAGE_v2_prep/storage_facility/numpyStore_BaseCase_slice_{}.npy'.format(beginID), allow_pickle=True)
baseMiddle = np.load('/scratch/orie3748/GraphSAGE_v2_prep/storage_facility/numpyStore_BaseCase_slice_{}.npy'.format(middleID), allow_pickle=True)
baseEnd = np.load('/scratch/orie3748/GraphSAGE_v2_prep/storage_facility/numpyStore_BaseCase_slice_{}.npy'.format(endID), allow_pickle=True)

print("Are the subreddits the same in the beginning slice?", flush=True)
beginSubsList = baseBegin[:,14].tolist()
print('\t',sorted(beginSubsList) == sorted(beginDictSubs), flush=True)
print("Is beginSubsList alphabetised already?", flush=True)
print('\t', beginSubsList == sorted(beginSubsList), flush=True)
print("\nAnd for the middle slice?", flush=True)
midSubsList = baseMiddle[:,14].tolist()
print('\t',sorted(midSubsList) == sorted(midDictSubs), flush=True)
print("Is midSubsList alphabetised already?", flush=True)
print('\t', midSubsList == sorted(midSubsList), flush=True)
print("\nAnd for the end slice?", flush=True)
endSubsList = baseEnd[:,14].tolist()
print('\t',sorted(endSubsList) == sorted(endDictSubs), flush=True)
print("Is endSubsList alphabetised already?", flush=True)
print('\t', endSubsList == sorted(endSubsList), flush=True)

# This gives dictionaries for each slice that give the subreddit and its corresponding row in the numpy Arrays (base and concept-specific.)

# Very possible just to use list.index method, but becomes cumbersome later as it is O(n) vs O(1) here.
beginPairMap = {}
for idx, sub in enumerate(beginSubsList):
    beginPairMap[sub] = idx

midPairMap = {}
for idx, sub in enumerate(midSubsList):
    midPairMap[sub] = idx

endPairMap = {}
for idx, sub in enumerate(endSubsList):
    endPairMap[sub] = idx


# Make 3 empty stores for new feature vectors in Numpy
featsBegin = np.empty(shape=(len(beginSubsList),7), dtype=object)
featsMid = np.empty(shape=(len(midSubsList),7), dtype=object)
featsEnd = np.empty(shape=(len(endSubsList),7), dtype=object)

rawFeatsBegin = np.empty(shape=(len(beginSubsList),7), dtype=object)
rawFeatsMid = np.empty(shape=(len(midSubsList),7), dtype=object)
rawFeatsEnd = np.empty(shape=(len(endSubsList),7), dtype=object)

# Loop over concepts
conceptsName = concepts_file.split('=')[1]
with open(conceptsName,'r') as cPTR:
    for line in cPTR:
        concept_list.append(line.split(':')[0])

print("The concepts, in order are:", concept_list, flush=True)

for conIdx, myConcept in enumerate(concept_list):
    # For each concept, open the numpy array for the files.
    fileStr = '/scratch/orie3748/GraphSAGE_v2_prep/storage_facility/numpyStore_' + myConcept + '_slice_{}' + '.npy'
    begStr = fileStr.format(beginID)
    midStr = fileStr.format(middleID)
    endStr = fileStr.format(endID)

    begin_mat = np.load(begStr, allow_pickle=True)
    mid_mat = np.load(midStr, allow_pickle=True)
    end_mat = np.load(endStr, allow_pickle=True)

    # Do the necessary normalisations/logs etc and add to the main numpy array
    # Start by looping over all subreddits for begin slice
    #for i in range(0,len(beginSubsList)):
        # For each subreddit, set the value of the featsBegin to be the value of the begin_mat divided by the baseArray
    #    beginVal = np.sum(begin_mat[i,:14])
    #    if beginVal == 0:
    #        beginVal += 1
    #    beginTotVal = np.sum(baseBegin[i,:14])
    #    featsBegin[i][conIdx] = math.log(beginVal/beginTotVal)
    #    featsBegin[i][6] = begin_mat[i,14]

    featsBegin, rawFeatsBegin = normaliser(beginSubsList, begin_mat, baseBegin, featsBegin, conIdx, rawFeatsBegin)
    featsMid, rawFeatsMid = normaliser(midSubsList, mid_mat, baseMiddle, featsMid, conIdx, rawFeatsMid)
    featsEnd, rawFeatsEnd = normaliser(endSubsList, end_mat, baseEnd, featsEnd, conIdx, rawFeatsEnd)

    # Repeat for all 6 concepts, so each of the 3 slices has now a normalised, log numpy matrix. Return those
np.save('numPy_condensed_{}'.format(beginID), featsBegin)
np.save('numPy_condensed_{}'.format(middleID), featsMid)
np.save('numPy_condensed_{}'.format(endID), featsEnd)

    # Then develop this by considering the difference features.
    # For each pair of slices, find the intersection of those subreddits.
#featsMid_Extend =
for i in [0,1]:
    if i == 0:
        myIntersect = set(beginSubsList) & set(midSubsList)
        myIntersect = sorted(list(myIntersect.copy()))
    else:
        secondIntersect = set(midSubsList) & set(endSubsList)
        secondIntersect = sorted(list(secondIntersect.copy()))

featsMid_Extend = np.empty(shape=(len(myIntersect),13), dtype=object)
featsEnd_Extend = np.empty(shape=(len(secondIntersect), 13), dtype=object)
# first label set: two numpy arrays for each of the slice intersection pairs
label_mid_intersect_init = np.empty(shape=(len(myIntersect),7), dtype=object)
label_end_intersect_init = np.empty(shape=(len(secondIntersect),7), dtype=object)

# Now the dictionaries from earlier come in handy.
# These are: beginPairMap, midPairMap and endPairMap
# These give the row of appearance for each subreddit in the NumPy arrays.

# We also need new dictionaries that give ordering in extended arrays
myInterDict = {}
secondInterDict = {}

for mIdx, j in enumerate(myIntersect):
    myInterDict[j] = mIdx
    featsMid_Extend[mIdx][12] = j
    label_mid_intersect_init[mIdx][0] = j
    featsMid_Extend[mIdx][:6] = featsMid[midPairMap[j]][:6]
    for k in range(6,12):
        # This is because the same subreddit can appear on different rows of NumPy matrices in different slices
        # Therefore we use the same subreddit but index different dictionaries for each respective numpy array
        # -6 because featsMid and featsBegin only go up to 6.
        featsMid_Extend[mIdx][k] = featsMid[midPairMap[j]][k - 6] - featsBegin[beginPairMap[j]][k - 6]
        label_mid_intersect_init[mIdx][k - 6] = normer(rawFeatsMid, rawFeatsBegin, k - 6, midPairMap, beginPairMap, j, pUp)

    label_mid_intersect_init[mIdx][6] = j

for mIdx, j in enumerate(secondIntersect):
    secondInterDict[j] = mIdx
    featsEnd_Extend[mIdx][12] = j
    featsEnd_Extend[mIdx][:6] = featsEnd[endPairMap[j]][:6]
    for k in range(6,12):
        featsEnd_Extend[mIdx][k] = featsEnd[endPairMap[j]][k - 6] - featsMid[midPairMap[j]][k - 6]
        label_end_intersect_init[mIdx][k - 6] = normer(rawFeatsEnd, rawFeatsMid, k - 6, endPairMap, midPairMap, j, pUp)

    label_end_intersect_init[mIdx][6] = j

    # Then make new numpy arrays for those, take existing values over as they should be.
    # Then add the difference elements. Return those too (x2)

np.save('numPy_diff_{}'.format(middleID), featsMid_Extend)
np.save('numPy_diff_{}'.format(endID), featsEnd_Extend)

np.save('numPy_diff_LABEL_{}'.format(middleID), label_mid_intersect_init)
np.save('numPy_diff_LABEL_{}'.format(endID), label_end_intersect_init)
    # Then finally for intersection of all 3

# For the intersection of all 3,
triple_intersect = set(beginSubsList) & set(midSubsList) & set(endSubsList)
triple_intersect = sorted(list(triple_intersect.copy()))
featsMid_triple = np.empty(shape=(len(triple_intersect),13), dtype=object)
featsEnd_triple = np.empty(shape=(len(triple_intersect),13), dtype=object)

label_mid_intersect_fin = np.empty(shape=(len(triple_intersect), 7), dtype=object)
label_end_intersect_fin = np.empty(shape=(len(triple_intersect), 7), dtype=object)


# Need to use myInterDict and secondInterDict to help index final arrays
for mIdx, j in enumerate(triple_intersect):
    featsMid_triple[mIdx][:] = featsMid_Extend[myInterDict[j]][:]
    featsEnd_triple[mIdx][:] = featsEnd_Extend[secondInterDict[j]][:]
    for y in range(0,6):
        label_mid_intersect_fin[mIdx][y] = normer(rawFeatsMid, rawFeatsBegin, y, midPairMap, beginPairMap, j, pUp)
        label_end_intersect_fin[mIdx][y] = normer(rawFeatsEnd, rawFeatsMid, y, endPairMap, midPairMap, j, pUp)

    # New change !!!!
    label_mid_intersect_fin[mIdx][6] = j
    label_end_intersect_fin[mIdx][6] = j


# finally save it
np.save('numPy_finDiff_{}'.format(middleID), featsMid_triple)
np.save('numPy_finDiff_{}'.format(endID), featsEnd_triple)

np.save('numPy_finDiff_LABEL_{}'.format(middleID), label_mid_intersect_fin)
np.save('numPy_finDiff_LABEL_{}'.format(endID), label_end_intersect_fin)


# For label generation (we'll do it for the final one alone.)
