import re #regular expressions are used to tokenize the text
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords #NLTK library is used to remove the stopwords in the corpus
import spacy #spacy library is used for text lemmatization
nlp = spacy.load('en')

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

        return finalProcessedText

    """
    Calculates the cosine similarity with sets T and B
    """
    def calculateCosineSimilarity(self):
        pass

    """
    Calculate the feature matrix T x B using the context window size 5 (two positions before and two after the target word). Use point-wise mutual information scores as weights.
    """
    def calculateFeatureMatix(self):
        pass

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