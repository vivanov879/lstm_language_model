from data_utils import utils as du
import pandas as pd

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)
##
# Below needed for 'adj_loss': DO NOT CHANGE
fraction_lost = float(sum([vocab['count'][word] for word in vocab.index
                           if (not word in word_to_num) 
                               and (not word == "UUUNKKK")]))
fraction_lost /= sum([vocab['count'][word] for word in vocab.index
                      if (not word == "UUUNKKK")])
print "Retained %d words from %d (%.02f%% of all tokens)" % (vocabsize, len(vocab),
                                                             100*(1-fraction_lost))

# Load the training set
docs = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs, word_to_num)
X_train, Y_train = du.seqs_to_lmXY(S_train)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs, word_to_num)
X_dev, Y_dev = du.seqs_to_lmXY(S_dev)

# Load the test set (final evaluation only)
docs = du.load_dataset('data/lm/ptb-test.txt')
S_test = du.docs_to_indices(docs, word_to_num)
X_test, Y_test = du.seqs_to_lmXY(S_test)

# Display some sample data
print " ".join(d[0] for d in docs[7])
print S_test[7]



with open('inv_vocabulary_raw', 'w') as f1:
    with open('vocabulary_raw', 'w') as f2:
        lines1 = []
        lines2 = []
        for word, num in word_to_num.items():
            lines1.append(word + ' ' + str(num+1) + '\n')
            lines2.append(str(num+1) + ' ' + word + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

with open('x_train', 'w') as f1:
    with open('y_train', 'w') as f2:
        lines1 = []
        lines2 = []
        for i in range(len(X_train)):
            lines1.append(' '.join([str(k+1) for k in X_train[i]]) + '\n')
            lines2.append(' '.join([str(k+1) for k in Y_train[i]]) + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

with open('x_dev', 'w') as f1:
    with open('y_dev', 'w') as f2:
        lines1 = []
        lines2 = []
        for i in range(len(X_dev)):
            lines1.append(' '.join([str(k+1) for k in X_dev[i]]) + '\n')
            lines2.append(' '.join([str(k+1) for k in Y_dev[i]]) + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

