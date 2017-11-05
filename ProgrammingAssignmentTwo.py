import re #regular expressions are used to tokenize the text
from math import log

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords #NLTK library is used to remove the stopwords in the corpus
import spacy #spacy library is used for text lemmatization
nlp = spacy.load('en')
import TwoDimensionalDictionary
class ProgrammingAssignmentTwo():

    """
    Globabl variables
    """
    textFile = None #this variable refers to the file text.txt
    basisFile = None #this variable refers to the file B.txt
    targetFile = None #this variable refers to the file T.txt
    rawText = "" #this variable will contain the raw text of text.txt
    processedText = [] #this list will contain all the words of text.txt file after processing
    basisWords = [] #this list will contain all the words of B.txt file
    targetWords = [] #this list will contain all the words of T.txt file
    wordsContingencyTable = {} # in order to get PMI weightnings we need to create a contingency table
    frequencyMatrix = TwoDimensionalDictionary.tdd()
    featureMatrix = TwoDimensionalDictionary.tdd()
    freqDict = {}

    """
    Class constructor
    Used for:
    * creating ProgrammingAssignmentTwo objects
    * initialization of global variables
    """
    def __init__(self, textFile, basisFile, targetFile):

        """
        :param textFile: text file (corpus)
        :param basisFile: basis words
        :param targetFile: target words
        """

        #Initialization of global variables
        self.textFile = textFile
        self.basisFile = basisFile
        self.targetFile = targetFile


    """
    readTextFile - reads a text file and return it's content as a string
    
    :returns self.rawText #content of text.txt
    :type string
    """
    def readTextFile(self):
        self.rawText = self.textFile.read()
        return self.rawText

    """
    readBasis - reads B.txt (the basis words file) and return it's content as a list of words
    
    :returns basisWords
    :type list
    """
    def readBasis(self):
        #read from B.txt
        rawBasis = self.basisFile.read()

        #append each word to the list
        for word in rawBasis.split('\n'):
            self.basisWords.append(word)

        return self.basisWords


    """
    readTerms- reads T.txt (the terms words file) and return it's content as a list of words
    
    :returns targetWords
    :type list
    """
    def readTarget(self):
        #read from T.txt
        rawTarget = self.targetFile.read()

        #append each word to the list
        for word in rawTarget.split('\n'):
            self.targetWords.append(word)

        return self.targetWords

    """
    textProcessing- performs the following operations to a string: 
    
    * separate words by white spaces, 
    * lowercase, 
    * remove all the punctuation, 
    * remove all the stop words from the current NLTK stop-word list,
    * lemmatise with 'spacy'
    
    :return processedText
    :type list
    """
    def textProcessing(self):
        processedText = []
        finalProcessedText = []

        #split by whitespaces, punctuation removal, lowercasing
        try:
            # Tokenize each line in the corpus, taking all the words and ignoring the punctuation using regular expression.
            # Append each word to the processedText list.
            # Word order will not be affected.
            for line in self.rawText.split('\n'):
                #white space separation and punctuation removal
                tokenizedLine = re.findall(r"[\w]+", line)
                # Append each token of the tokenized line to processedText list
                for token in tokenizedLine:
                    processedText.append(token.lower()) #lowercasing
        except Exception as e:
            print("An error occured during text processing.")
            print(e)

        #stopwords removal
        try:
            for word in stopwords.words('english'):
                for processedTextWord in processedText:
                    if(word == processedTextWord):
                        processedText.remove(processedTextWord)
        except Exception as e:
            print("An error occured during stopwords removal.")
            print(e)
        #for debugging purposes, print the english stopwords
        #print(stopwords.words('english'))

        #lemmatization
        try:
            processedTextString = ""
            for word in processedText:
                processedTextString += word + " "
            lemmatizedText = nlp(processedTextString)

            for token in lemmatizedText:
                #For debug: see detalied lemmas
                #print(token, token.lemma, token.lemma_)
                finalProcessedText.append(token.lemma_)
        except Exception as e:
            print("An error occured during lemmatization.")
            print(e)

        self.processedText = finalProcessedText
        return finalProcessedText

    """
    Create frequency matrix
    """
    def createFrequencyMatrix(self):
        frequencyMatrix = {}
        processedTextLength = len(self.processedText)-1

        #creating a dictionart of tuples that is, first word represents a row entry(a target word), and the second one represents the column entry(a basis word)
        #record their frequency
        for target in self.targetWords:
            for basis in self.basisWords:
                pairOfWords = tuple((target, basis))
                frequencyMatrix[pairOfWords] = 0


        for i in range(0,processedTextLength):
            if (self.processedText[i] in self.targetWords):
                #print(self.processedText[i] + " appears in the target words list\n") #for debug purposes

                # treat special cases when the window cannot be exactly -2 w +2
                if (i < 2):
                    if(i == 0):
                        if (self.processedText[i + 1] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i + 1]))
                            frequencyMatrix[pairOfWords] += 1
                        if (self.processedText[i + 2] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i + 2]))
                            frequencyMatrix[pairOfWords] += 1
                    if(i == 1):
                        if (self.processedText[i - 1] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i - 1]))
                            frequencyMatrix[pairOfWords] += 1
                        if (self.processedText[i + 1] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i + 1]))
                            frequencyMatrix[pairOfWords] += 1
                        if (self.processedText[i + 2] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i + 2]))
                            frequencyMatrix[pairOfWords] += 1

                elif (i > processedTextLength - 2):
                    if(i == processedTextLength-1):
                        if (self.processedText[i - 2] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i - 2]))
                            frequencyMatrix[pairOfWords] += 1
                        if (self.processedText[i - 1] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i - 1]))
                            frequencyMatrix[pairOfWords] += 1
                        if (self.processedText[i + 1] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i + 1]))
                            frequencyMatrix[pairOfWords] += 1
                    if(i == processedTextLength):
                        if (self.processedText[i - 2] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i - 2]))
                            frequencyMatrix[pairOfWords] += 1
                        if (self.processedText[i - 1] in self.basisWords):
                            pairOfWords = tuple((self.processedText[i], self.processedText[i - 1]))
                            frequencyMatrix[pairOfWords] += 1

                else:
                    if (self.processedText[i - 2] in self.basisWords):
                        pairOfWords = tuple((self.processedText[i], self.processedText[i - 2]))
                        frequencyMatrix[pairOfWords] += 1
                    if (self.processedText[i - 1] in self.basisWords):
                        pairOfWords = tuple((self.processedText[i], self.processedText[i - 1]))
                        frequencyMatrix[pairOfWords] += 1
                    if (self.processedText[i + 1] in self.basisWords):
                        pairOfWords = tuple((self.processedText[i], self.processedText[i + 1]))
                        frequencyMatrix[pairOfWords] += 1
                    if (self.processedText[i + 2] in self.basisWords):
                        pairOfWords = tuple((self.processedText[i], self.processedText[i + 2]))
                        frequencyMatrix[pairOfWords] += 1

        #For debug purposes
        #print(frequencyMatrix) #for debug purposes

        for tuples, frequencies in frequencyMatrix.items():
            self.frequencyMatrix[str(tuples[0])][str(tuples[1])] = frequencies

        self.freqDict = frequencyMatrix

        #For debug purposes
        #print(self.frequencyMatrix)
        #print(self.frequencyMatrix['lady']['lucas'])
        return frequencyMatrix

    """
    Calculates the cosine similarity with sets T and B
    """
    def calculateCosineSimilarity(self):
        pass


    """
    Creation of contingency table in order to create: A, B, C, D, R1, R2, C1, C2, N
    N.B.: this method is a helper method for: calculateFeatureMatix
    """
    def createContigencyTable(self, word1, word2):
        contingencyTable = []
        # Contigency table variables
        a = 0
        b = 0
        c = 0
        d = 0
        r1 = 0
        r2 = 0
        c1 = 0
        c2 = 0
        n = sum(self.freqDict.values())

        for tuple, val in self.freqDict.items():
            if (word1 == tuple[0] and word2 == tuple[1]):
                a += val
            elif (word1 == tuple[0] and word2 != tuple[1]):
                b += val
            elif (word1 != tuple[0] and word2 == tuple[1]):
                c += val

        d = n - a - b - c

        n = a + b + c + d
        r1 = a + b
        c1 = a + c

        contingencyTable = [a, b, c, d, n, r1, c1]
        return contingencyTable

    """
    Calculate PMI (MI score)
    N.B.: this method is a helper method for: calculateFeatureMatix
    """

    def calculatePMI(self, contingencyTable):

        (a, b, c, d, n, r1, c1) = contingencyTable

        a += 1
        b += 1
        c += 1
        d += 1
        n += 4

        r1 = a + b
        c1 = a + c

        return log(float(a) / (float(r1) * float(c1)) * float(n), 2)

    """
    Calculate the feature matrix T x B using the context window size 5 (two positions before and two after the target word). Use point-wise mutual information scores as weights.
    """
    def calculateFeatureMatix(self):

        for tuples, freq in self.freqDict.items():

                #print(tuples, freq)
                targetWord = tuples[0]
                basisWord = tuples[1]
                self.featureMatrix[targetWord][basisWord] = self.calculatePMI(self.createContigencyTable(targetWord,basisWord))

        return self.featureMatrix

    """
    Calculate the cosine similarity matrix T x T using the PMI feature matrix.
    """
    def calculateCosineSimilarityMatrixTT(self):
        pass

    """
    Convert the similarity score into distance. The output of this step are two matrices: one for similarity, one for distance.
    """
    def convertSimilarityScoreIntoDistance(self):
        pass

    """
    Group the most similar words together using two functions prepared by Tatyana:  hierarchical clustering and k-means.
    """
    def createClusters(self):
        pass


"""
Preparing the files:
textFile => corpus
basisFile => basis words file
targetFile => target words file
"""
textFile = open("text.txt", "r", encoding="UTF-8")
basisFile = open("B.txt", "r", encoding="UTF-8")
targetFile = open("T.txt", "r", encoding="UTF-8")

pa2Obj = ProgrammingAssignmentTwo(textFile, basisFile, targetFile)


#Debugging helppers
print(pa2Obj.readTextFile())
print(pa2Obj.readBasis())
print(pa2Obj.readTarget())
print(pa2Obj.textProcessing())
print(pa2Obj.createFrequencyMatrix())
print(pa2Obj.calculateFeatureMatix())