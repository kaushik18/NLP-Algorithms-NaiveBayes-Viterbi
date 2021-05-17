# Kaushik Nadimpalli
# CS6320 Homework 2
# Part 2 - POS Tagging with Naive Bayes (using provided formula and txt file for training)
  # Tip:  Use the WORD_POS pattern to extract the actual word and part-of-speech tag (i.e. the WORD part in the WORD_POS
  # pattern is the actual word and POS part in the WORD_POS pattern is the part-of-speech tag) from the tokenized word.
  # For example, in the tokenized word “Brainpower_NNP”, “Brainpower” is the actual word and “NNP” is the part-of-speech tag.
  # No tokenization needed
  # No smoothing needed

import sys
import itertools
import re

def main():
    path = sys.argv[1]
    inputfile = open(path, "r")
    string = inputfile.read()
    count_of_tags = dict()

    for l in string.splitlines():
        l = l.strip()
        count_of_tags["<s>"] = 1 + count_of_tags.get("<s>", 0)
        for word in l.split():
            tag = word.split("_")[1]
            count_of_tags[tag] = 1 + count_of_tags.get(tag, 0)
        count_of_tags["</s>"] = 1 + count_of_tags.get("</s>", 0)

    word_count_of_tags=dict()
    c_tags=dict()
    wordtags=dict()

    for l in string.splitlines():
        words = l.strip().split()
        word = words[0].split("_")[0]
        tag = words[0].split("_")[1]
        word_count_of_tags[tag,word] = 1 + word_count_of_tags.get((tag,word), 0)
        c_tags["<s>",tag] = 1 + c_tags.get(("<s>",tag), 0)
        if word not in wordtags.keys():
            wordtags[word] = set()
            wordtags[word].add(tag)
        else:
            wordtags[word].add(tag)

        for node in range(1,len(words)-1):
            tag_previous = tag
            word = words[node].split("_")[0]
            tag = words[node].split("_")[1]
            word_count_of_tags[tag,word] = 1 + word_count_of_tags.get((tag,word), 0)
            c_tags[tag_previous,tag] = 1 + c_tags.get((tag_previous,tag), 0)
            if word not in wordtags.keys():
                wordtags[word] = set()
                wordtags[word].add(tag)
            else:
                wordtags[word].add(tag)

        # part of speech tag assignment
        tag_previous = tag
        word = words[-1].split("_")[0]
        tag = words[-1].split("_")[1]
        word_count_of_tags[tag,word] = 1 + word_count_of_tags.get((tag,word), 0)
        c_tags[tag_previous,tag] = 1 + c_tags.get((tag_previous,tag), 0)
        c_tags[tag,"</s>"] = 1 + c_tags.get((tag,"</s>"), 0)

        if word not in wordtags.keys():
            wordtags[word] = set()
            wordtags[word].add(tag)
        else:
            wordtags[word].add(tag)

    word_tag_prob = dict()
    probability_with_assign_tag = dict()
    for k in word_count_of_tags:
        word_tag_prob[k] = word_count_of_tags[k] / count_of_tags[k[0]]
    for k in c_tags:
        probability_with_assign_tag[k] = c_tags[k] / count_of_tags[k[0]]

    print("Naive Bayes POS Tagging for Input sequence")
    user_sen = sys.argv[2]
    user_sen_words = []

    for w in user_sen.split():
        user_sen_words.append(w)
    words_tags = []

    for w in user_sen_words:
        words_tags.append(list(wordtags[w]))

  # Core functionality of applying probability calculaton using Naive Bayes formula
    probability_maximum = 0
    partofspeech_tags = None
    for sequence in itertools.product(*words_tags):
        probability = 1
        for x in range(len(user_sen_words)):
            if x == 0:
                if ("<s>",sequence[x]) not in probability_with_assign_tag.keys():
                    probability=0
                    break
                else:
                    probability *= word_tag_prob[sequence[x],user_sen_words[x]] * probability_with_assign_tag[("<s>",sequence[x])]
            elif x == len(user_sen_words) - 1:
                if (sequence[x-1],sequence[x]) not in probability_with_assign_tag.keys() or (sequence[x],"</s>") not in probability_with_assign_tag.keys():
                    probability = 0
                    break
                else:
                    probability *= word_tag_prob[sequence[x],user_sen_words[x]]* probability_with_assign_tag[sequence[x-1],sequence[x]] * probability_with_assign_tag[sequence[x],"</s>"]
            else :
                if (sequence[x-1],sequence[x]) not in probability_with_assign_tag.keys():
                    probability = 0
                    break
                else:
                    probability *= word_tag_prob[sequence[x],user_sen_words[x]]*probability_with_assign_tag[sequence[x-1],sequence[x]]
        if probability > probability_maximum:
            probability_maximum = probability
            partofspeech_tags = sequence

    print("Words in Input Sentence")
    print()
    print("\t",user_sen_words)
    print()
    print("Tags - Assigned to Sentence")
    print()
    print("\t",list(partofspeech_tags))
    print()
    print()
    print("Tag Sequence Probability")
    print("\t",probability_maximum)
    print()

if __name__ == '__main__':
    main()
