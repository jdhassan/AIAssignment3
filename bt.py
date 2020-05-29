# Bayesian Tomatoes:
# Doing some Naive Bayes and Markov Models to do basic sentiment analysis.
#
# Input from train.tsv.zip at
# https:#www.kaggle.com/c/sentiment-analysis-on-movie-reviews
#
# itself gathered from the Rotten Tomatoes movie review aggregation site.
#
# Format is PhraseID[unused]   SentenceID  Sentence[tokenized] Sentiment
#
# We'll only use the first line for each SentenceID, since the others are
# micro-analyzed phrases that would just mess up our counts.
#
# Sentiment is on a 5-point scale:
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
#
# For each kind of model, we'll build one model per sentiment category.
# Following Bayesian logic, base rates matter for each category; if critics
# are often negative, that should be a good guess in the absence of other
# information.
#
# Training input is assumed to be in a file called "train.tsv"
#
# Test sentences are received via stdin (thus either interactively or with input redirection).
# Output for each line of input is the following:
#
# Naive Bayes classification (0-4)
# Naive Bayes most likely class's log probability (with default double digits/precision)
# Markov Model classification (0-4)
# Markov Model most likely class's log probability

import sys
import math

# NLTK is a great toolkit that makes Python a nice choice for natural language processing (NLP)
# ... although to use this tokenizer, you will first need to 
# >>> import nltk
# >>> nltk.download('punkt')
# ... to get the dataset it relies on
from nltk.tokenize import word_tokenize
# Note that bigrams() returns an *iterator* and not a *list* - you will need to call it
# again or store as a list to iterate over the bigrams multiple times
from nltk.util import bigrams

CLASSES = 5
# Assume sentence numbering starts with this number in the file

FIRST_SENTENCE_NUM = 1
# Probability of either a unigram or bigram that hasn't been seen -
# needs to be small enough that it's "practically a rounding error"
OUT_OF_VOCAB_PROB = 0.0000000001

# tokenize():  get words from a line in a consistent way
# uses NLTK, standardizes to lowercase
# returns list of tokens
def tokenize(sentence):
    return [t.lower() for t in word_tokenize(sentence)]


class ModelInfo:
    # ModelInfo fields:
    # word_counts:  5 dicts from string to int, one per sentiment, in a list
    # bigram_counts: same
    # sentiment_counts:  list containing counts of the sentences with different sentiments
    # total_words:  list of counts of words for each sentiment
    # bigram_denoms:  Separate counts of how often a word starts a bigram, again one per sentiment.
    #   (Not quite the same as the word count since last words of sentences aren't counted here)
    # total_bigrams: counts of total bigrams for each sentiment level

    def __init__(self):
        self.word_counts = [{}, {}, {}, {}, {}]
        self.bigram_counts = [{}, {}, {}, {}, {}]
        self.sentiment_counts = [0, 0, 0, 0, 0]
        self.total_words = [0, 0, 0, 0, 0]
        self.bigram_denoms = [{}, {}, {}, {}, {}]
        self.total_bigrams = [0, 0, 0, 0, 0]
        self.total_examples = 0


    # update_word_counts
    # assume space-delimited words/tokens.
    #
    # To "tokenize" the sentence we'll make use of NLTK, a widely-used Python natural language
    # processing (NLP) library.  This will handle otherwise onerous tasks like separating periods
    # from their attached words.  (Unless the periods are decimal points ... it's more complex
    # than you might think.)  The result of tokenization is a list of individual strings that are
    # words or their equivalent.
    #
    # Note that sentiment is an integer, not a string, matching the data format
    def update_word_counts(self, sentence, sentiment):
        # Get the relevant dicts for the sentiment
        s_word_counts = self.word_counts[sentiment]
        s_bigram_counts = self.bigram_counts[sentiment]
        s_bigram_denoms = self.bigram_denoms[sentiment]
        tokens = tokenize(sentence)
        for token in tokens:
            self.total_words[sentiment] += 1
            s_word_counts[token] = s_word_counts.get(token, 0) + 1
        my_bigrams = bigrams(tokens)
        for bigram in my_bigrams:
            s_bigram_counts[bigram] = s_bigram_counts.get(bigram, 0) + 1
            s_bigram_denoms[bigram[0]] = s_bigram_denoms.get(bigram[0], 0) + 1
            self.total_bigrams[sentiment] += 1

# get_models:  returns a model_info object
def get_models():
    next_fresh = FIRST_SENTENCE_NUM
    info = ModelInfo()
    for line in sys.stdin:
        if line.startswith("---"):
            return info
        fields = line.split("\t")
        try:
            sentence_num = int(fields[1])
            if sentence_num != next_fresh:
                continue
            next_fresh += 1
            sentiment = int(fields[3])
            info.sentiment_counts[sentiment] += 1
            info.total_examples += 1
            info.update_word_counts(fields[2], sentiment)
        except ValueError:
            # Some kind of bad input?  Unlikely with our provided data
            continue
    return info

# classify_sentences:  takes a ModelInfo, reads sentences from stdin, prints
# their classification info for the two models
def classify_sentences(info):
    for line in sys.stdin:
        nb_class, nb_logprob = naive_bayes_classify(info, line)
        mm_class, mm_logprob = markov_model_classify(info, line)
        print(nb_class)
        print(nb_logprob)
        print(mm_class)
        print(mm_logprob)            

# naive_bayes_classify:  takes a ModelInfo containing all counts necessary for classsification
# and a String to be classified.  Returns a number indicating sentiment and a log probability
# of that sentiment (two comma-separated return values).
def naive_bayes_classify(info, sentence):
    sentiment_decision = 0
    max_log_probability = -float("inf")

    sentiment = 0
    while (sentiment < CLASSES):
        word_counts = info.word_counts.__getitem__(sentiment)

        sentiment_words = count_sentiment_words(word_counts)

        log_probability = math.log(info.sentiment_counts[sentiment]) - math.log(count_sentences(info.sentiment_counts))

        for s in sentence.split(" "):
            s = s.lower()
            word_count = word_counts.get(s)
            if word_count == None or sentiment_words == 0:
                log_probability += math.log(OUT_OF_VOCAB_PROB)
            else:
                log_probability += math.log(word_count) - math.log(sentiment_words)
        
        if (log_probability > max_log_probability):
            sentiment_decision = sentiment
            max_log_probability = log_probability

        sentiment += 1

    return sentiment_decision, max_log_probability

    '''

        //calculates the log probability for each sentiment P(sentiment | sentence) which is
        //proportional to the sum from i = 1 to n P(word at i | sentiment) + P(sentiment)
        for (int sentiment = 0 ; sentiment < CLASSES ; sentiment++) {

            //The word counts for this sentiment
            Map<String, Integer> wordCounts = info.wordCounts.get(sentiment);

            //the total number of words in this sentiment
            double sentimentWords = sentimentWords(wordCounts);

            //initializes the lob probability to the probability of the sentiment P(sentiment),
            //which is defined by (# of sentences with sentiment / # of sentences)
            double logProbability = Math.log(info.sentimentCounts[sentiment]) - Math.log(sentences(info.sentimentCounts));

            //calculates P(word | sentiment) for all words in the sentence, and adds them to logProb
            for (String s : sentence.split(" ")) {
                s = s.toLowerCase();
                //the number of times word 's' appears for the sentiment
                Integer wordCount = wordCounts.get(s);
                if (wordCount == null || sentimentWords == 0) {
                    logProbability += Math.log(OUT_OF_VOCAB_PROB);
                } else {
                    //Adds P(word | sentiment) = wordCount / # of sentiment words, to log probability
                    logProbability += Math.log(wordCount) - Math.log(sentimentWords);
                }
            }

            //updates maxLogProbability if the logProbability for the current sentiment is greater
            if (logProbability > maxLogProbability) {
                sentimentDecision = sentiment;
                maxLogProbability = logProbability;
            }
        }

        return new Classification(sentimentDecision, maxLogProbability);
        '''
    #return 0, 0 # best class, log probability
    
# markov_model_classify:  like naive Bayes, but now use a bigram model.  First word
# still uses unigram count & probability.
def markov_model_classify(info, sentence):
    # TODO
    return 0, 0 # best class, log probability

def count_sentiment_words(counts):
    count = 0
    for i in counts.values():
        count += i
    return count

def count_sentences(sentences):
    count = 0
    for i in sentences:
        count += i
    return count

if __name__ == "__main__":
    info = get_models()
    classify_sentences(info)
