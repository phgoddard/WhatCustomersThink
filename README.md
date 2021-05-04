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
with open('/users/tandemseven/desktop/3 python/ms.txt',encoding='mac-roman') as f:
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
