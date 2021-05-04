## Welcome to the Tech Project

This tutorial explores the use of word and document embeddings as a means to search and find customer comments with similar content. Here are the basic steps in the tutorial.

![2021-05-04_12-07-46](https://user-images.githubusercontent.com/70239535/117056625-63f3da80-acd1-11eb-8c63-1071cfcd672f.png)

### The downloads used in this tutorial
Here are the basic python resources used in the tutorial:

```markdown
import re
from math import sqrt
import spacy
nlp = spacy.load('en_core_web_sm')
import pke
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('universal_tagset')
stop_words = stopwords.words('english')
import numpy as np
from scipy.spatial import distance
```
### Step 1. Load Customer comments and generate keyphrases
For this step I found a keyphrase module for python called 'pke'.
The documentation had a great pipeline developed using nltk and spaCy.
```markdown
# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using spacy
extractor.load_document(input='../path to data/..txt')
                        
import string
# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
k = extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=30)

# Sample Output  
0 ('newworld', 0.03727223506914867)
1 ('newworld partner network', 0.022279041000711687)
2 ('product', 0.012087648493261223)
3 ('times', 0.011867073939491941)
4 ('support', 0.010576520470561054)
5 ('difficult', 0.010282139581529043)
6 ('business', 0.010157070647847495)
7 ('customer', 0.009308289405822616)
8 ('office', 0.007764396498753803)
9 ('licensing', 0.0075731972343565665)
10 ('halve year', 0.007156489593132479)
```
The output shows the top 10 ranked keyphrases with their ranking.
I added indexes to tuples for reference.

### Step 2. Preprocess Customer Comments
This step includes reading the comments file and preprocessing.  I use spacy's nlp pipeline for tokenization, the re module for normalization and text cleaning. After reviewing the output I found some empty comments that needed to be removed.  The output is a grand list of comments.

```markdown
with open('../path to data/..txt',encoding='mac-roman') as f:
    lines = [line.strip().split('\n') for line in f.readlines()];
    
#run each response through spacy to preprocess and tokenize
toks=[]
sents=[]
i = 0
while i < len(lines):
    doc = nlp(lines[i][0])
    for tok in doc:    
        t = tok.text.lower()       #lowercase all
        t = re.sub('•','',t)       #remove unwanted marks, blanks
        t = re.sub('-*$|^-*','',t) #remove dashes 
        toks.append(t)
    sents.append(toks)
    toks=[]
    i+=1

#test for and remove blank sentences
nsents = []
print("original # sentences: ", len(sents))

for i,sent in enumerate(sents):
    if len(sent) == 0 :
        print("line",i,"found blank")
    else:
        nsents.append(sent)

print("count with blank removed: ",len(nsents))

#Sample pre-processed comment

nsents[4]
['licensing', 'and', 'no', 'support', 'for', 'older', 'technology', '.']
```
### Step 3. Load Pre-trained word embeddings
This step included downloading the GloVe word embeddings found here: https://nlp.stanford.edu/projects/glove/.
I used the file: glove.6B.50d.txt

```markdown
I found a convenience function to load the embeddings into my jupyter notebook:

def load_glove_vectors(fn):
    print("Loading Glove Model")
    with open( fn, 'r', encoding='utf8') as glove_vector_file:
        model = {}
        for line in glove_vector_file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print("Loaded {} words".format(len(model)))
    return model
    
glove_vectors = load_glove_vectors('.../path/glove.6B/glove.6B.50d.txt')

#the output
Loading Glove Model
Loaded 400000 word

#Once you've loaded the embeddings, it's fun to see what's there - here's one you've probably seen before.
glove_vectors['dog']
array([ 0.11008  , -0.38781  , -0.57615  , -0.27714  ,  0.70521  ,
        0.53994  , -1.0786   , -0.40146  ,  1.1504   , -0.5678   ,
        0.0038977,  0.52878  ,  0.64561  ,  0.47262  ,  0.48549  ,
       -0.18407  ,  0.1801   ,  0.91397  , -1.1979   , -0.5778   ,
       -0.37985  ,  0.33606  ,  0.772    ,  0.75555  ,  0.45506  ,
       -1.7671   , -1.0503   ,  0.42566  ,  0.41893  , -0.68327  ,
        1.5673   ,  0.27685  , -0.61708  ,  0.64638  , -0.076996 ,
        0.37118  ,  0.1308   , -0.45137  ,  0.25398  , -0.74392  ,
       -0.086199 ,  0.24068  , -0.64819  ,  0.83549  ,  1.2502   ,
       -0.51379  ,  0.04224  , -0.88118  ,  0.7158   ,  0.38519  ])

```
```markdown
# Header 1
## Header 2
### Header 3
- Bulleted
- List
1. Numbered
2. List
**Bold** and _Italic_ and `Code` text
[Link](url) and ![Image](src)
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
```
