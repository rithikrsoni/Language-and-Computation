import math
from collections import defaultdict, Counter
from treebank import train_corpus, test_corpus, conllu_corpus
from sys import float_info
from math import log, exp



class BigramLanguage: 

    """
    Initilises Bigram Language model.
    
    """
    def __init__(self, z= 10**4):
        self.bigramCounts = defaultdict(Counter)
        self.unigramCounts = Counter()
        self.totalWordCount = 0
        self.z = z # smoothing factor

    
    def train(self,corpus):
        """
        Function to train the bigram model by counting the occurrences of word sequences in the corpus. 
        adds <s> to represent start of sentence </s> end of sentence. 
        """
        for sentence in corpus:
            prevWord = "<s>"
            self.unigramCounts[prevWord] += 1 

            for token in sentence:
                word = token['form']
                self.bigramCounts[prevWord][word] += 1 
                self.unigramCounts[word] += 1
                self.totalWordCount += 1
                prevWord = word

            self.bigramCounts[prevWord]["</s>"] += 1
            self.unigramCounts["</s>"]+= 1

    """
    Witten Bell Function 
    """
    def wittenBell(self, counts, totalCount, totalUnique):
        if totalCount == 0: 
            return{key:1 / self.totalWordCount for key in self.unigramCounts}
        
        lambdat = totalCount / (totalCount+ totalUnique)
        smoothedProbability ={}
        
        for word, count in counts.items():
            bigramProb = count / totalCount
            unigramProb = self.unigramCounts[word] / self.totalWordCount if word in self.unigramCounts else 1/ self.totalWordCount
            smoothedProbability[word] = lambdat * bigramProb + (1 - lambdat) * unigramProb

        

       
        unseenProb = totalUnique / ((self.z + totalCount) * totalUnique) if totalUnique > 0 else 1e-10
        smoothedProbability['<UNK>'] = unseenProb
        
        return smoothedProbability

    """
    Bigram Probability 
    """
    def computeBigramProbs(self):

        self.bigramProbs = {} 
        for prevWord, counts in self.bigramCounts.items():
            totalCount = self.unigramCounts[prevWord]
            totalUnique = len(counts)
            self.bigramProbs[prevWord] = self.wittenBell(counts,totalCount, totalUnique)

    
    """
    Perplexity function 
    """
    
    def perplexity(self,testData):

        logProbSum = 0
        totalLength = 0 
        

        for sentence in testData:
            prevWord = "<s>"
            for token in sentence: 
                word = token['form']
                prob = self.bigramProbs.get(prevWord,{}).get(word,self.bigramProbs.get(prevWord,{}).get('<UNK>', 1e-10))
                logProbSum +=math.log(prob,2)
                totalLength += 1
                prevWord = word

            prob = self.bigramProbs.get(prevWord,{}).get("</s>", self.bigramProbs.get(prevWord, {}).get('<UNK>', 1e-10))
            logProbSum += math.log(prob,2)
            totalLength +=1
        
        return 2 ** (-logProbSum / totalLength) if totalLength > 0 else float('inf')
    
    """
    Evaluation of the model. Comparing actual predictions to expected predictions and then 
    """
    def accuracy(self,testData):
        correct, total = 0,0
        
        for sentence in testData:
            prevWord =" <s>"

            for token in sentence: 
                word = token["form"].lower()
                total +=1
                
                if prevWord in self.bigramProbs:
                    predictedWord = max(self.bigramProbs[prevWord], key=self.bigramProbs[prevWord].get)
                else: 
                    predictedWord = "<UNK>"

                if predictedWord == word:
                    correct += 1

                prevWord = word 

        return correct / total if total > 0 else 0.0
        
    

languages = ['en', 'orv', 'tr']


for lang in languages:
    print(f"\nTraining bigram model for: {lang}")
    trainData = conllu_corpus(train_corpus(lang))
    testData = conllu_corpus(test_corpus(lang))
    
    bigramModel = BigramLanguage()
    bigramModel.train(trainData)
    bigramModel.computeBigramProbs()
    
    bigramAccuracy = bigramModel.accuracy(testData)
    perplexity = bigramModel.perplexity(testData)
   
    print(f"Bigram Accuracy: {bigramAccuracy:.4f}") 
    print(f"Perplexity for {lang}: {perplexity:.4f}")