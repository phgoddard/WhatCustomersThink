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
extractor.load_document(input='/Users/tandemseven/desktop/3 python/MS.txt'
                        , language='en', encoding='mac-roman')
                        
import string
# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
k = extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=30)
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
