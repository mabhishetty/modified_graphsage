# modified_graphsage

#### Introduction

- Code for the GraphSAGE algorithm (https://arxiv.org/pdf/1706.02216.pdf) 
- Forked from https://github.com/williamleif/graphsage-simple, then edited
- Original algorithm aims to learn node-level representations through propagation of representations from nodes in a neighbourhood.
- My use case:
  - I have set up networks to represent social structure of Reddit. Between communities on Reddit, I want to see how accurately the spread of COVID19-related concepts can be predicted.
    - Nodes: Subreddits
    - Edges: Hypothesised links between subreddits based on proportion of commenters that two subreddits share in common.
  - There are 12 networks that discretise a six-month period of Reddit data into 2-week slices (each network is a snapshot of a 2-week slice of Reddit)
  - For each node, I have features that represent the frequency with which a set of 6 concepts - related to Coronavirus - are discussed. This feature vector is specific to that subreddit, in that particular 2-week time slice.
  - My aim is to use GraphSAGE to predict, for each node in a given graph, whether the frequency of discussion of a concept in that subreddit (node) has increased or not from a previous time-slice.

#### Code structure

- **go1** contains the edited Python scripts: this is where I imagine a bug might lie.
- The algorithm works by calling model.py first. 
  - First, this sets up the graphs, features and labels.
  - Then a SupervisedGraphSage (neural net) object is instantiated.
  - The dataset is partitioned into training and testing/validation
  - Training proceeds on samples of the training set (stopping criterion = # iterations at the moment. Later, will add overfitting prevention by checking loss on val periodically)
    - Within training: model.py -> encoders.py
    - encoders.py generates embeddings for a set of nodes (it has its own weight matrix that gets optimised in training)
    - encoders.py -> aggregators.py
    - aggregators.py takes features from a nodes neighbourhood. Meant to consider sampling too. 
    - Once these other calls have finished, model.py gives, for each node input on an iteration, a matrix with scores for each class-subreddit prediction
    - Then xentropy loss is used (includes a softmax) to give loss
    - **As far as I can tell, the only two learnt parameters during backprop are: weight matrix in model.py used for producing class-subreddit prediction scores and a weight matrix in encoders.py when generating embeddings**
  - Then the validation set is used to give an F1 score.

- **data_store** contains all the data go1 needs to run on. There are two graphs, for testing and training (s5 is for training and s6 is meant for test/val). There are feature and label NumPy arrays too (dtype=object - hence allow_pickle=True - here because subreddit names are stored within)
- **requirementsNEW.txt** contains all the Python modules that were installed in my virtualenv on ARC when I ran these jobs. Not all can be installed using conda, some need pip
- **assembly_gsage_feats_gsage2.py** has the code I used to generate the feature vectors. The input feature matrices have 14-element feature vectors for each node, where the elements of the vector correspond to the actual frequency of a given concept on a day in the 2-week slice.
  
  These are modified to form a 12-element feature vector for each node. The first 6 elements are total frequencies of each concept at that node over the 2 week period (so summing over a feature vector in the input Numpy matrix). The last 6 elements are the *changes* in frequencies of those concepts from the last slice.
  (eg: [freq_of_concept #1 freq_con_#2 freq_con_#3 freq_con_#4 freq_con_#5 freq_con_#6 change_in_freq_con_#1 change_in_freq_con_#2 change_in_freq_con_#3 change_in_freq_con_#4 change_in_freq_con_#5 change_in_freq_con_#6]

#### The problem

- The output files are in **go1** (extensions: .out and .err)
- The F1 score at the moment is very low - but I don't think that the model is training correctly/ideally.
- As in **go1/gsage_2.out**, the loss hardly changes after iteration 3 which is not ideal.
- The average batch time is incredibly low and the whole algorithm takes less than a minute to run.
  - The algorithm was originally designed to support ~O(10^5) nodes and took a few days on their datasets.
  - Not sure if the runtime is due to my small graphs or a bug (?)
- Trying to figure out why it isn't training properly (loss isn't really decreasing after iterations)

#### Areas of investigation

- Is loss not decreasing because of a (local) minimum in the optimisation?
- What is happening that the loss is going so high (iteration 1/2)?
- I assume the loss decreases drastically because the gradient calculated is so sharp?
- Why does the loss not change between iteration 3 and 100?
- Learning rate too high?
- Poor random initialisation?
- Is the aggregation right (mean aggs, should the gcn=True?)
- Is the sampling correct? I don't think so. What about no sampling? Doesn't seem to have sampling with replacement properly implemented?
- What is the structure of these new networks? Are they in many large/split components?
- Should I just try the largest connected component instead? / can I get largest connected component of this new bit?
- Dimensions of embeddings generated?
- Why is it running so fast?
- Non-log methods? Maybe the non-linearity is messing things up? Though I think there is a non-linearity in the model so should be able to represent it? 
