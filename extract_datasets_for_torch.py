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


