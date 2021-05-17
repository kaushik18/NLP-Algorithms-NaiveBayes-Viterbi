# Kaushik Nadimpalli
# CS6320 Homework 2
# Part 2 - viterbi_alg Algoritm

# Test examples
  # i. Janet will back the bill
  # ii. will Janet back the bill
  # iii. back the bill Janet will

import sys

user_sen = sys.argv[1]
tag_states = ['NNP','MD','VB','JJ','NN','RB','DT']
likelihood = []
words = user_sen.split(" ")
for w in words:
    likelihood.append(w)

# The HMM observation likelihood is also hardcoded below according to the provided probabilities
# store in dict and pass in nxn matrix format
hmm_obs_likelihood = dict()
hmm_obs_likelihood['NNP'] = {'Janet':0.000032,'will':0,'back':0,'the':0.000048,'bill':0}
hmm_obs_likelihood['MD'] = {'Janet':0,'will':0.308431,'back':0,'the':0,'bill':0}
hmm_obs_likelihood['VB'] = {'Janet':0,'will':0.000028,'back':0.000672,'the':0,'bill':0.000028}
hmm_obs_likelihood['JJ'] = {'Janet':0,'will':0,'back':0.000340,'the':0,'bill':0}
hmm_obs_likelihood['NN'] = {'Janet':0,'will':0.000200,'back':0.000223,'the':0,'bill':0.002337}
hmm_obs_likelihood['RB'] = {'Janet':0,'will':0,'back':0.010446,'the':0,'bill':0}
hmm_obs_likelihood['DT'] = {'Janet':0,'will':0,'back':0,'the':0.506099,'bill':0}

# The HMM transition probabilities are hardcoded below for the different transition tag_states
# store in dict and pass in nxn matrix format
tran_prob = dict()
tran_prob['<s>'] = {'NNP':0.2767,'MD':0.0006,'VB':0.0031,'JJ':0.0453,'NN':0.0449,'RB':0.0510,'DT':0.2026}
tran_prob['NNP'] = {'NNP':0.3777,'MD':0.0110,'VB':0.0009,'JJ':0.0084,'NN':0.0584,'RB':0.0090,'DT':0.0025}
tran_prob['MD'] = {'NNP':0.0008,'MD':0.0002,'VB':0.7968,'JJ':0.0005,'NN':0.0008,'RB':0.1698,'DT':0.0041}
tran_prob['VB'] = {'NNP':0.0322,'MD':0.0005,'VB':0.0050,'JJ':0.0837,'NN':0.0615,'RB':0.0514,'DT':0.2231}
tran_prob['JJ'] = {'NNP':0.0366,'MD':0.0004,'VB':0.0001,'JJ':0.0733,'NN':0.4509,'RB':0.0036,'DT':0.0036}
tran_prob['NN'] = {'NNP':0.0096,'MD':0.0176,'VB':0.0014,'JJ':0.0086,'NN':0.1216,'RB':0.0177,'DT':0.0068}
tran_prob['RB'] = {'NNP':0.0068,'MD':0.0102,'VB':0.1011,'JJ':0.1012,'NN':0.0120,'RB':0.0728,'DT':0.0479}
tran_prob['DT'] = {'NNP':0.1147,'MD':0.0021,'VB':0.0002,'JJ':0.2157,'NN':0.4744,'RB':0.0102,'DT':0.0017}

# Implementation of Viterbi Algorithm

 # Create an array
  #  With columns corresponding to inputs  Rows corresponding to possible tag_states
  # Sweep through the array in one pass filling the columns left to right using our transition probs and likelihood probs
  # Dynamic programming key is that we need only store the MAX prob path to each cell, (not all paths) and tracebacking.

def viterbi_alg(hmm_obs_likelihood,tran_prob):

    # Initialization
    trace_back = dict()
    viterbi_alg = dict()
    viterbi_alg["begin"] = []

    # Recursion

    for s in tag_states:
        viterbi_alg[s] = []
        trace_back[s] = dict()
        viterbi_alg[s].append(tran_prob['<s>'][s] * hmm_obs_likelihood[s][likelihood[0]])
        trace_back[s][0] = ('begin', tran_prob['<s>'][s] * hmm_obs_likelihood[s][likelihood[0]])

    # Recursive step of calling algorithm
    for x in range(1,len(likelihood)):
        for s in tag_states:
            prev_max = ''
            counter = -1
            for prev_s in tag_states:
                if counter < viterbi_alg[prev_s][x-1] * tran_prob[prev_s][s]:
                    counter = viterbi_alg[prev_s][x-1] * tran_prob[prev_s][s]
                    prev_max = prev_s
            viterbi_alg[s].append(0)
            viterbi_alg[s][x] = counter * hmm_obs_likelihood[s][likelihood[x]]
            trace_back[s][x] = (prev_max,counter)
    counter = -1
    max_s = ''

    for s in tag_states:
        if counter < viterbi_alg[s][-1]:
            counter = viterbi_alg[s][-1]
            max_s = s

    # Termination
    trace_back['A'] = dict()
    trace_back['A'][len(likelihood)] = (max_s, counter)
    begin = 'A'
    sequence_prob = 1 # initial value
    tags = [] # store the tags
    for x in range(len(likelihood),0,-1):
        if x == len(likelihood):
            sequence_prob = trace_back[begin][x][1]
            tags.insert(0,trace_back[begin][x][0])
        else:
            tags.insert(0, trace_back[begin][x][0])
        begin = trace_back[begin][x][0]

  # Output formatting and probability printint
    sentence_tag = []
    for i in range(len(tags)):
        sentence_tag.append(likelihood[i]+'_'+ tags[i])
    print()
    print("User Sentence - with assigned tags")
    print()
    print("\t",sentence_tag)
    print()
    print()
    print("Observation Sequence Probability")
    print()
    print("\t",sequence_prob)
    print()
    print()

if __name__ == '__main__':
    # The hardcoded probabilites in instructions are passed as parameters to do the prob calculations
    viterbi_alg(hmm_obs_likelihood,tran_prob)
