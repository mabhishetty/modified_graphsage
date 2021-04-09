# issues: xavier_uniform vs xavier_uniform_
# Need to change 0 indexing in labels to 6.

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

# my bits -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
import sys
from sys import argv
import networkx as nx
import pandas as pd
import math
#-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
# Edited: 26/03/21
# Editing Author: MA (original: W. Hamilton)
"""


class SupervisedGraphSage(nn.Module):

    # num_classes: Different classes for classification task (no. of)
    # enc: Greatest level encoder used
    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        # Final encoder
        self.enc = enc
        # Loss function for supervised training.
        self.xent = nn.CrossEntropyLoss()
        # Tensor that has a row for each class, and the number of columns is the dimension of the final node representation. DIM: [nc x finalEmbedDim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        # Initialise with values chosen uniformly. # FUNCTION BELOW WAS CHANGED #
        init.xavier_uniform(self.weight)

        # Making a prediction.
        # nodes: nodes on which to predict
    def forward(self, nodes):
        # Generate a suggested representation for these nodes. Since the final enc is used (drawing on other enc's), dim(embeds) == dim(enc.embed_dim) above
        # DIM: [finalEmbedDim x nNodes] (?)
        embeds = self.enc(nodes)
        # Multiply the weight matrix by the embeddings. This generates the predictions. DIM: [nc x nNodes] (because: [nc x finalEmbedDim] * [finalEmbedDim x nNodes])
        scores = self.weight.mm(embeds)
        # Returns [nNodes x nc]: for each node has a row vector, where the elements are the scores for each class?
        return scores.t()

        # Finding the loss on the predicted nodes
    def loss(self, nodes, labels):
        # return the
        scores = self.forward(nodes)
        # From the PyTorch docs, https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html, the input should have dimension (minibatch, C)
        # This is compatible with [nNodes x nc]. In the docs: loss(x,C), I think 'x' represents the tensor associated with a single node.
        # Then loss is averaged over all x's, all nodes in the minibatch.
        # Input: (N, C) where C = number of classes, N: number of nodes in minibatch
        #        Target: (N): no. nodes in minibatch, with the true labels
        return self.xent(scores, labels.squeeze())

def load_reddit(whole_graph_name, theMap):
    # -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/

    # Only important bit here - that feat_data and labels have the same ordering. That's all that node_map seeks to do.

    # Going to try to keep the structure as similar as possible to the original for as long as possible.
    # Hence I will use features from all the nodes
    #num_nodes_tot = len(allNodes)
    #num_feats = dimensionFeatures

    #print("Total number of nodes: ", num_nodes_tot, flush=True)

    # Hence, all labels (and all features) are in a single data structure.
    #labels = np.empty((num_nodes_tot,1), dtype=np.int64)

    # -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/

    adj_lists = defaultdict(set)
    allNodes = list(whole_graph_name.nodes)
    print("Double-checking that we have the same nodes from the graph as in the features/labels.", flush=True)
    print("Is this true?", sorted(allNodes) == sorted(list(theMap.keys())), flush=True)
    for node in allNodes:
        # want to gather the links for each subreddit
        linksTuples = list(whole_graph_name.edges([node]))
        links = set([theMap[j[1]] for j in linksTuples])
        # sets are mutable!!
        adj_lists[theMap[node]] = links.copy()

    return adj_lists

def run_reddit(ovr_graph_name, dim_feats, featuresArray, labelsArray):
    np.random.seed(1)
    random.seed(1)

    # -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
    # want a list of all the networkx nodes
    #nodelist_G = list(ovr_graph_name.nodes)
    # split up the nodes of the overall graph. Still have one graph, in disjoint components. This is because some nodes are repeated between the two distinct sections

    # Want to get features for all nodes - train and test. So pass all nodes as argument at this point.
    print("Loading Reddit data now...", flush=True)
    # Not going to bother passing features/labels in here. All that the function does is return them, and I already have them.
    # KEY SETUP:
    # Will need a key also. At this point, both the labels and the features array ought to have the same ordering. So I will do something with this
    # Trains and tests are always going to be separate because of the concat step. Within each, the subs may/may not be alphabetised
    theListOfFeatNames = featuresArray[:,dim_feats].tolist()
    theMapping = {}
    for indx, name in enumerate(theListOfFeatNames):
        theMapping[name] = indx
    # Now we have a mapping for all the subreddits (where trains are ALWAYS from 0->: and tests follow. Within each, they may not be alphabetised.)
    # Importantly, the indx values correspond not only to subreddits (via hash function), but also = ROWS of those subreddits in featuresArray (and by extension, labelsArray.)

    print("The mapping has {} keys".format(len(list(theMapping.keys()))), flush=True)

    adj_lists = load_reddit(ovr_graph_name, theMapping)

    print("Done loading Reddit data. Continuing with model...", flush=True)

    num_nodes = len(list(ovr_graph_name.nodes))
    features = nn.Embedding(num_nodes, dim_feats)
    # Recall that featuresArray includes not only the features, but also the name of the subreddits as the 15th column [:,14]
    # Need to remove this in order to convert numpy array to integer for weight part here
    # Will need to repeat with the labels

    featuresArray = featuresArray[:,:dim_feats]
    print("Updated feature shape:", featuresArray.shape, flush=True)
    featuresArray = featuresArray.astype(np.int64)
    features.weight = nn.Parameter(torch.FloatTensor(featuresArray), requires_grad=False)

    # For this one, the names were stored in the 0th column
    labelsArray = labelsArray[:,0]
    print("Updated label shape:", labelsArray.shape, flush=True)
    labelsArray = labelsArray.astype(np.int64)

    # features.cuda()

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, dim_feats, dim_feats, adj_lists, agg1, gcn=True, cuda=False)


     # agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
     # enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
     #         base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 25
     #enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc1)

    train_node_names = []
    test_and_val_names = []
    for node in theListOfFeatNames:
        if node.endswith('_train'):
            train_node_names.append(node)
        elif node.endswith('_test'):
            test_and_val_names.append(node)
        else:
            print("Problem when assembling indices before model run...", flush=True)
            sys.exit()

    print("The number of training nodes we have - which should equal training numbers from earlier - is:", len(train_node_names), flush=True)
    print("The number of test and val nodes we have is:", len(test_and_val_names), flush=True)

    train = [theMapping[node] for node in train_node_names]
     # this generates random ordering of integers including 0 and up to - but not including - len(test_and_val_names). But len(test_and_val_names) won't be a valid index, so we are fine.
     # These indices are just integers - but eventually they should correspond to rows in the data stores
     # Start just by randomising the test_and_val_names list
    rand_indices_v_t = np.random.permutation(len(test_and_val_names))
     # from the paper, 30% for validation
    numVal = int(math.floor(0.3*len(test_and_val_names)))
     # This gives us the indices for validation nodes
    valIndices = list(rand_indices_v_t[:numVal])
     # indices for test nodes
    testIndices = list(rand_indices_v_t[numVal:])

     # Now acquire those actual node indices - first node names, then use those to get indices
    valNodes = [theMapping[test_and_val_names[j]] for j in valIndices]
    testNodes = [theMapping[test_and_val_names[k]] for k in testIndices]

    val = np.array(valNodes, dtype=np.int64)
    test = np.array(testNodes, dtype=np.int64)
    #-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/


    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(labelsArray[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item(), flush=True)

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labelsArray[val], val_output.data.numpy().argmax(axis=1), average="micro"), flush=True)
    print("Average batch time:", np.mean(times), flush=True)


def new_order(feature_np, label_np, featDim):

    feature_frame = pd.DataFrame(data=feature_np)
    label_frame = pd.DataFrame(data=label_np)

    sorted_feature_frame = feature_frame.sort_values(by=[featDim])
    sorted_label_frame = label_frame.sort_values(by=[0])

    sorted_feature_numpy = sorted_feature_frame.to_numpy(dtype=object, copy=True)
    sorted_label_numpy = sorted_label_frame.to_numpy(dtype=object, copy=True)

    sortedFeaturesList = sorted_feature_numpy[:,featDim].tolist()
    sortedLabelList = sorted_label_numpy[:,0].tolist()

    print("Is the ordering fixed?", sortedFeaturesList == sortedLabelList, flush=True)

    return sorted_feature_numpy, sorted_label_numpy



def graph_feat_label_prep(graphTRAIN, graphTEST, featTRAIN, featTEST, labelTRAIN, labelTEST, numFeats, concepts_path, concept_q):
    """ This function takes:
        Input: Train graph, test graph, train feats, test feats, train labels, test labels and number of features.
        Output: combined graph, combined features and combined labels.
    """
    # UPDATE: in v2, confident that features and labels are all alphabetised already. Same order already.
    # First reading in features
    trainFeats = np.load(featTRAIN, allow_pickle=True)
    testFeats = np.load(featTEST, allow_pickle=True)

    print("Number of rows in training features:", trainFeats.shape[0], flush=True)
    print("Number of rows in test features:", testFeats.shape[0], flush=True)

    trainLabels = np.load(labelTRAIN, allow_pickle=True)
    testLabels = np.load(labelTEST, allow_pickle=True)

    print("Number of rows in training labels:", trainLabels.shape[0], flush=True)
    print("Number of rows in test labels:", testLabels.shape[0], flush=True)

    trainNodesNumpyFeatList = trainFeats[:,numFeats].tolist()
    testNodesNumpyFeatList = testFeats[:,numFeats].tolist()
    trainNodesNumpyLabelList = trainLabels[:,6].tolist()
    testNodesNumpyLabelList = testLabels[:,6].tolist()

    # Now checking equality of lists and hence ordering
    # I don't like how this looks either...
    trainCheck = trainNodesNumpyFeatList == trainNodesNumpyLabelList
    testCheck = testNodesNumpyFeatList == testNodesNumpyLabelList


    print("Is the ordering of training the same?:", trainCheck , flush=True)
    print("Is the ordering of testing the same?:", testCheck, flush=True)

    if trainCheck == False:
        print("The training ordering is messed up. Fixing it...", flush=True)
        trainFeats, trainLabels = new_order(trainFeats, trainLabels, numFeats)
        print("The training ordering should be fixed.", flush=True)

    if testCheck == False:
        print("The testing ordering is messed up. Fixing it...", flush=True)
        testFeats, testLabels = new_order(testFeats, testLabels, numFeats)
        print("The testing ordering should be fixed.", flush=True)

    ######

    print("Are the node lists the same? We have already checked each feat against its label one. Checking across train and test now...", flush=True)
    print(trainNodesNumpyFeatList == testNodesNumpyFeatList, flush=True)
    print("Are these sorted too?", flush=True)
    print(sorted(trainNodesNumpyFeatList) == trainNodesNumpyFeatList, flush=True)

    print("Features read in. Reading in graphs now...", flush=True)

    ######

    # read graphs in
    trainGRAPH = nx.read_gexf(graphTRAIN, version='1.2draft')
    testGRAPH = nx.read_gexf(graphTEST, version='1.2draft')
    # relabelling dictionary
    #trainDICT = {}
    #testDICT = {}
    trainNODElist = list(trainGRAPH.nodes)
    testNODElist = list(testGRAPH.nodes)
    print("The number of nodes in the training graph is:", len(trainNODElist), flush=True)
    print("The number of nodes in the test list is:", len(testNODElist), flush=True)

    # Need to check if nodes have degree 0
    # BUT... cannot check for degree 0 in original graph. Need to check between nodes that are all included.
    # Because, if degree == 1 and the node to which the edge was ended up being removed, degree == 0.
    # So need to make subgraph first.
    # In reality, these lists are the same. Even the trainNodesNumpyFeatList and testNodesNumpyFeatList are the same.
    trainInclusionList = []
    testInclusionList = []

    trainOrderDict = {}
    testOrderDict = {}

    # Removes all nodes that are not in the triple intersection.
    # In total, removing 2 types of nodes. Non-triple intersect ones and ones with 0 degree.

    trainInitSubgraph = trainGRAPH.subgraph(trainNodesNumpyFeatList).copy()
    testInitSubgraph = testGRAPH.subgraph(testNodesNumpyFeatList).copy()


    # same sequence of subreddits
    for i in range(0,len(trainNodesNumpyFeatList)):
        sub = trainFeats[i,numFeats]
        trainOrderDict[sub] = i
        if trainInitSubgraph.degree[sub] > 0:
            trainInclusionList.append(sub)

    for j in range(0, len(testNodesNumpyFeatList)):
        sub = testFeats[j, numFeats]
        testOrderDict[sub] = j
        if testInitSubgraph.degree[sub] > 0:
            testInclusionList.append(sub)

    trainTracerList = trainInclusionList.copy()
    testTracerList = testInclusionList.copy()

    # But there are issues with the procedure so far. Nodes that have non-zero degree in one slice may have zero degree elsewhere.
    # This means that their removal will impact the degree of nodes in the 1st graph. It could lead to more nodes having degree 0.
    # Try iterations to show that we are ok here.

    #This condition is False iff NO NODES are found that are degree 0 in one slice and degree >0 in another.
    # because if we have the same set of nodes it means we have a set of nodes whose degree >0 and are in both.
    counterA = 0
    while trainTracerList != testTracerList:
        counterA += 1
        # List of nodes that currently have degree > 0 in both slices. We could get changes in structure because, for a given slice, some of its key nodes might not be in the other slice
        intList = sorted(list(set(trainTracerList) & set(testTracerList)).copy())
        # make from the original graph.
        trainInterimGraph = trainGRAPH.subgraph(intList).copy()
        testInterimGraph = testGRAPH.subgraph(intList).copy()

        trainTracerList = []
        testTracerList = []

        for i in intList:
            if trainInterimGraph.degree[i] > 0:
                trainTracerList.append(i)
            if testInterimGraph.degree[i] > 0:
                testTracerList.append(i)

    finalTrainList = trainTracerList.copy()
    finalTestList = testTracerList.copy()

    finalTrainGraph = trainInterimGraph.copy()
    finalTestGraph = testInterimGraph.copy()

    print("Iterations of this:", counterA, flush=True)
    print("Are degree > 0 lists the same?", flush=True)
    print(finalTrainList == finalTestList, flush=True)
    print("Are they sorted (degree > 0 lists?)", flush=True)
    print(finalTrainList == sorted(finalTrainList), flush=True)
    print("Length of degree > 0 lists is:", len(finalTrainList), flush=True)
    # Now we have lists of subreddits for test and train that have degree > 0 (and appear in both graphs.)
    # They should be the same
    # Graph composition

    final_train_dict = {}
    final_test_dict = {}

    for node in finalTrainList:
        final_train_dict[node] = node + '_train'

    for node in finalTestList:
        final_test_dict[node] = node + '_test'

    nx.relabel_nodes(finalTrainGraph, final_train_dict, copy=False)
    nx.relabel_nodes(finalTestGraph, final_test_dict, copy=False)

    combo_graph = nx.compose(finalTrainGraph, finalTestGraph)
    print("The number of nodes in the composition is:", len(list(combo_graph.nodes)), flush=True)

    # For features and labels (new ones)
    concept_list = []
    with open(concepts_path,'r') as cPTR:
        for line in cPTR:
            concept_list.append(line.split(':')[0])

    print("The concepts, in order are:", concept_list, flush=True)
    print("The concept of choice is:", concept_q, flush=True)
    print("This is at index: {} in the list".format(concept_list.index(concept_q)), flush=True)

    newTrainFeats = np.empty(shape=(len(finalTrainList), (numFeats + 1)), dtype=object)
    newTrainLabels = np.empty(shape=(len(finalTrainList), 2), dtype=object)

    newTestFeats = np.empty(shape=(len(finalTestList), (numFeats + 1)), dtype=object)
    newTestLabels = np.empty(shape=(len(finalTestList), 2), dtype=object)

    # Making new feature matrices
    for idx, sub in enumerate(finalTrainList):
        newTrainFeats[idx][numFeats] = final_train_dict[sub]
        newTrainFeats[idx][:numFeats] = trainFeats[trainOrderDict[sub]][:numFeats]

        newTrainLabels[idx][1] = final_train_dict[sub]
        newTrainLabels[idx][0] = trainLabels[trainOrderDict[sub]][concept_list.index(concept_q)]

    for idx, sub in enumerate(finalTestList):
        newTestFeats[idx][numFeats] = final_test_dict[sub]
        newTestFeats[idx][:numFeats] = testFeats[testOrderDict[sub]][:numFeats]

        newTestLabels[idx][1] = final_test_dict[sub]
        newTestLabels[idx][0] = testLabels[testOrderDict[sub]][concept_list.index(concept_q)]

    combo_feats = np.concatenate((newTrainFeats,newTestFeats))

    print("Shape of updated feature array (train + test):", combo_feats.shape, flush=True)

    combo_labels = np.concatenate((newTrainLabels,newTestLabels))

    print("Shape of updated label array (train + test):", combo_labels.shape, flush=True)

    print("Are the final feature orderings the same?", flush=True)
    print(combo_feats[:,numFeats].tolist() == combo_labels[:,1].tolist(), flush=True)

    return combo_graph, combo_feats, combo_labels


    # Now made the new feature and label matrices. Remember this function will returned combined: graphs, feature and label matrices.

    #for trainNode in trainNODElist:
#        trainDICT[trainNode] = trainNode + '_train'

    #for testNode in testNODElist:
#        testDICT[testNode] = testNode + '_test'

    # Changed the names of the nodes.
    #nx.relabel_nodes(trainGRAPH, trainDICT, copy=False)
    #nx.relabel_nodes(testGRAPH, testDICT, copy=False)

    # combine the two graphs into one
    #combo_graph = nx.compose(trainGRAPH, testGRAPH)
    #print("The number of nodes in the composition is:", len(list(combo_graph.nodes)), flush=True)

    # Work on the feature combination

    # First thing - !!!!!!!!!!!!!!!!!!!!!!!!!!!! SUPER IMPORTANT !!!!!!!! Ensure that features and labels have same ordering. If not, will need to reorder.

    # To begin, rename the subreddits as necessary


    # shape gives (#rows, #cols) tuple for a NumPy matrix. Range does not include the last number - but that's fine because we have 0-based indexing (ie, actual #rows wouldn't be a valid index)
    #for i in range(0,len(trainFeats[:,numFeats].tolist())):
#        trainFeats[i,numFeats] = trainFeats[i,numFeats] + '_train'

    #for i in range(0,len(testFeats[:,numFeats].tolist())):
    #    testFeats[i,numFeats] = testFeats[i,numFeats] + '_test'




    # Now for labels
    #for i in range(0,len(trainLabels[:,0].tolist())):
    #    trainLabels[i,0] = trainLabels[i,0] + '_train'

    #for i in range(0,len(testLabels[:,0].tolist())):
    #    testLabels[i,0] = testLabels[i,0] + '_test'


def fileNameGen(rawStr):
    return rawStr.split('=')[1].strip()

if __name__ == "__main__":
    #-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
    # the nodes will have names based on the graph they are from. So we can have the `same' node in two places.
    script_name, graph_1_name, graph_2_name, labels1_path, labels2_path, feature_dim, feature_path1, feature_path2, concepts_path, desired_concept = argv
    print("GraphSAGE first model. 1:1 graph prediction.", flush=True)
    print("Begin by prepping graphs...", flush=True)
    if not graph_1_name.startswith('test') or not graph_2_name.startswith('train'):
        print("Wrong format for entering graphs...", flush=True)
        sys.exit()

    if not labels1_path.startswith('test') or not labels2_path.startswith('train'):
        print("Wrong format for entering labels...", flush=True)
        sys.exit()

    if not feature_path1.startswith('test') or not feature_path2.startswith('train'):
        print("Wrong format for entering features...", flush=True)
        sys.exit()

    if not feature_dim.startswith('feature_dim'):
        print("Wrong format for entering feature dimensions...", flush=True)

    if not concepts_path.startswith('concepts='):
        print("wrong format for entering concepts...", flush=True)

    if not desired_concept.startswith('concept='):
        print("Wrong format for entering desired concept...", flush=True)

    graphTest = fileNameGen(graph_1_name)
    graphTrain = fileNameGen(graph_2_name)
    labelsTest = fileNameGen(labels1_path)
    labelsTrain = fileNameGen(labels2_path)
    featureTest = fileNameGen(feature_path1)
    featureTrain = fileNameGen(feature_path2)
    feature_dim = int(fileNameGen(feature_dim))
    concepts_P = fileNameGen(concepts_path)
    desired_concept_NAME = fileNameGen(desired_concept)

    # The outputs here are the combined test-train graph, test-train features and test-train labels. These are all that need changing in v2.
    combi_graph, combi_feats, combi_labels = graph_feat_label_prep(graphTrain, graphTest, featureTrain, featureTest, labelsTrain, labelsTest, feature_dim, concepts_P, desired_concept_NAME)
    print("Graphs prepped. Now running model...", flush=True)
    # Now the features and the labels should have the same ordering. If they were not previously ordered, they will now be ordered first by train/test, and within each of those - alphabetically.
    run_reddit(combi_graph, feature_dim, combi_feats, combi_labels)

    #-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
