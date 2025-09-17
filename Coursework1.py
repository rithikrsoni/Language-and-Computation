import math
from collections import defaultdict, Counter
from treebank import train_corpus, test_corpus, conllu_corpus
from sys import float_info
from math import log, exp

min_log_prob = -float_info.max # This helps prevent underflow. 

def logsumexp(vals): 
    if len(vals) == 0:
        return min_log_prob
    m = max(vals)
    if m == min_log_prob:
        return min_log_prob
    else:
        return m + log(sum([exp(val - m) for val in vals]))
    
    



class HiddenMarkovPOSTagger:
    def __init__(self, z= 10**4):
        """
        Initalise the HMM POS tagger.
        """
        self.transitionCounts = defaultdict(Counter)
        self.emissionCounts = defaultdict(Counter)
        self.tagCounts = Counter()
        self.wordCounts = Counter()
        self.z = z  # Smoothing factor. 
        self.tagSet = set()
        self.totalWordCount = 0

    def train(self, taggedSentences):
        
        """
        Train the model by counting transitions and emissions from tagged sentences. 
        adds <s> and </s> to represent start and end of token respectively. 
        """
        
        for sentence in taggedSentences:
            prevTag = '<s>'  # Start of sentence
            self.tagCounts[prevTag] += 1

            for token in sentence:  
                word = token['form']
                tag = token['upos']

                self.transitionCounts[prevTag][tag] += 1
                self.emissionCounts[tag][word] += 1
                self.tagCounts[tag] += 1
                self.wordCounts[word] += 1
                self.tagSet.add(tag)

                prevTag = tag  # Move to next tag

            self.transitionCounts[prevTag]['</s>'] += 1
            self.tagCounts['</s>'] += 1

    def wittenBell(self, counts, totalCount, totalUnique):
        """ 
        Compute Witten-Bell smoothed probabilities for both transition and emission cases.
        Returns:
        Dictionary of smoothed probabilities.
    """
        if totalCount == 0:
            return {key: 1 / sum(self.wordCounts.values()) for key in self.wordCounts}  # Uniform probability
    
    # Compute lambda wi-1 (backoff weight)
        lambdat = totalCount / (totalCount + totalUnique)

        smoothedProbability = {}

        for key, count in counts.items():
            bigramProb = count / totalCount  # P(wi | wi-1) or P(word | tag)
            unigramProb = self.tagCounts[key] / sum(self.tagCounts.values()) if key in self.tagCounts else 1 / sum(self.wordCounts.values())  # P(wi) or P(word)
            smoothedProbability[key] = lambdat * bigramProb + (1 - lambdat) * unigramProb 

    # Probability for the unseen 
        unseenProb = totalUnique / ( self.z * (totalCount + totalUnique)) if totalUnique > 0 else 1e-10
        smoothedProbability['<UNK>'] = unseenProb

        return smoothedProbability


    def computeTransitionProbs(self):
        """
        Compute transition probabilities using Witten-Bell smoothing 
        """
        self.transitionProbs = {}
    
        for prevTag, counts in self.transitionCounts.items():
            totalCount = self.tagCounts[prevTag]  
            totalUnique = len(counts)  
            self.transitionProbs[prevTag] = self.wittenBell(counts, totalCount, totalUnique)

    def computeEmissionProbs(self):
        """
        Computes emission probabilities using Witten-Bell smoothing for word-tag pairs.
        """
        self.emissionProbs = {}
        self.totalWordCount = sum(self.wordCounts.values())

        for tag, wordCounts in self.emissionCounts.items():
            tagCount = self.tagCounts[tag]  
            totalUnique = len(wordCounts)  
            self.emissionProbs[tag] = self.wittenBell(wordCounts, tagCount, totalUnique)

    def viterbi(self, sentence):
        """
        Viterbi algorithm for sequence decoding finding the most likely tag sequence.
        """
        
        if not sentence:
            return []

        V = [{}]
        path = {}

        for tag in self.tagSet:
            V[0][tag] = math.log(self.transitionProbs.get('<s>', {}).get(tag, 1e-10)) + \
                        math.log(self.emissionProbs.get(tag, {}).get(sentence[0], self.emissionProbs.get(tag, {}).get('<UNK>', 1e-10)))
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            newPath = {}

            for tag in self.tagSet:
                (prob, state) = max(
                    (V[t - 1][prevTag] + math.log(self.transitionProbs.get(prevTag, {}).get(tag, 1e-10)) +
                     math.log(self.emissionProbs.get(tag, {}).get(sentence[t], self.emissionProbs.get(tag, {}).get('<UNK>', 1e-10))), prevTag)
                    for prevTag in self.tagSet
                )
                V[t][tag] = prob
                newPath[tag] = path[state] + [tag]

            path = newPath

        (prob, state) = max(
            (V[len(sentence) - 1][tag] + math.log(self.transitionProbs.get(tag, {}).get('</s>', 1e-10)), tag)
            for tag in self.tagSet
        )
        return path[state]

    def evaluate(self, testData):
        correct = total = 0
        for sentence in testData:
            words = [token['form'] for token in sentence]
            trueTags = [token['upos'] for token in sentence]

            predictedTags = self.viterbi(words)

            correct += sum(p == t for p, t in zip(predictedTags, trueTags))
            total += len(trueTags)
        return correct / total if total > 0 else 0  # Avoid division by zero
    
    """
    Code for part 2 is below. The addition of the forward algorithm and the perplexity calculation. 

    Forward Algorithm function to compute the probability of an observation sequence.
    """
    def forwardAlgorithm(self, sentence): 
     
        if not sentence:
            return min_log_prob

        V = [{}]
        

        for tag in self.tagSet:
            V[0][tag] = math.log(self.transitionProbs.get('<s>', {}).get(tag, 1e-10)) + \
                        math.log(self.emissionProbs.get(tag, {}).get(sentence[0], 
                                                            self.emissionProbs.get(tag, {}).get('<UNK>', 1e-10)))
                            
        for t in range(1, len(sentence)):
            V.append({})
            for tag in self.tagSet: 
               prevLogProb = [
                   V[t - 1][prevTag] + log(self.transitionProbs.get(prevTag, {}).get(tag, 1e-10))
                   for prevTag in self.tagSet
               ]
               V[t][tag] = logsumexp(prevLogProb) + log(self.emissionProbs.get(tag, {}).get(sentence[t],
                                    self.emissionProbs.get(tag, {}).get('<UNK>' , 1e-10)))
                
        finalLogProb = [
            V[len(sentence) - 1][tag] + log(self.transitionProbs.get(tag,{}).get('</s>' , 1e-10))
            for tag in self.tagSet  
            ]
        return logsumexp(finalLogProb)
   
    def perplexity(self, testData):
        """
        Compute perplexity of the test corpus using the forward algorithm.
        where N is total tokens + sentence count.
        """
        logProbSum = 0
        totalLength = 0

        for sentence in testData:
            words = [token['form'] for token in sentence]
            logProb = self.forwardAlgorithm(words)  

        if logProb == min_log_prob:
            return float('inf')  # Avoid invalid probabilities

        logProbSum += logProb / math.log(2)  # Convert log to log base 2
        totalLength += len(words) + 1  # +1 for end-of-sentence token

        return 2 ** (-logProbSum / totalLength) if totalLength > 0 else float('inf')




langagesTest = ['en', 'orv', 'tr']

for lang in langagesTest:
    print(f"\nTraining model for: {lang}")
    trainData = conllu_corpus(train_corpus(lang))
    testData = conllu_corpus(test_corpus(lang))

    tagger = HiddenMarkovPOSTagger()
    tagger.train(trainData)
    tagger.computeTransitionProbs()
    tagger.computeEmissionProbs()
    
    perplexity = tagger.perplexity(testData)
    

    accuracy = tagger.evaluate(testData)
    print(f"Accuracy for {lang}: {accuracy:.4f}")
    print(f"Perplexity on test set: {perplexity:.4f}")