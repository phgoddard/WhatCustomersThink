## A technical tutorial using word embeddings to find similar customer comments
### by Phil Goddard

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

# Sample Output  (actually generated 30 keyphrases for last step above)
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
    
# run each response through spacy to preprocess and tokenize
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

# test for and remove blank sentences
nsents = []
print("original # sentences: ", len(sents))

for i,sent in enumerate(sents):
    if len(sent) == 0 :
        print("line",i,"found blank")
    else:
        nsents.append(sent)

print("count with blank removed: ",len(nsents))

# Sample pre-processed comment

nsents[4]
['licensing', 'and', 'no', 'support', 'for', 'older', 'technology', '.']
```
### Step 3. Load Pre-trained word embeddings and do some quick testing
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

# Once you've loaded the embeddings, it's fun to see what's there - here's one you've probably seen before.
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

# Run some tests on comments and keyphrases with embeddings to see how they work
I'm using word embeddings from glove on my customer comments and keyphrases so I need to test how well glove embeddings map.
As expected, not all words in my customer comments are included in glove, or customer comments as tokens were misspelled or short hand.
So I went through a process of converting customer comments and keyphrases to word embeddings, cleaning data further, and researching what to do when a word is not included in the glove vectors.

I ended up following a recommendation to use the 'unk' word embedding for unknown words, instead of setting them to zeros or something else.
Here's a code snippet of that:

# review all tokens not in glove_vectors (refine re step above)
UNKNOWN = []
for sent in nsents:
    for word in sent:
        if word not in glove_vectors:
            UNKNOWN.append(word)
print(set(UNKNOWN))

# in case you were curious, here's glove has for unk vector
glove_vectors['unk']
array([-7.9149e-01,  8.6617e-01,  1.1998e-01,  9.2287e-04,  2.7760e-01,
       -4.9185e-01,  5.0195e-01,  6.0792e-04, -2.5845e-01,  1.7865e-01,
        2.5350e-01,  7.6572e-01,  5.0664e-01,  4.0250e-01, -2.1388e-03,
       -2.8397e-01, -5.0324e-01,  3.0449e-01,  5.1779e-01,  1.5090e-02,
       -3.5031e-01, -1.1278e+00,  3.3253e-01, -3.5250e-01,  4.1326e-02,
        1.0863e+00,  3.3910e-02,  3.3564e-01,  4.9745e-01, -7.0131e-02,
       -1.2192e+00, -4.8512e-01, -3.8512e-02, -1.3554e-01, -1.6380e-01,
        5.2321e-01, -3.1318e-01, -1.6550e-01,  1.1909e-01, -1.5115e-01,
       -1.5621e-01, -6.2655e-01, -6.2336e-01, -4.2150e-01,  4.1873e-01,
       -9.2472e-01,  1.1049e+00, -2.9996e-01, -6.3003e-03,  3.9540e-01])

# sample of unknowns that I left till another day to deal with
{'', 'segawas', 'incentivizes', '............', 'molp', '   ', 'metalogix', 'ocina', '  ', 'crippleware', '204/2005', '07866777744', 'projetos', 'wgraph', 'sharepointonline', 'gdpr', 'i´m', 'microsource', }
```
### Step 4. Create doc embeddings as average from all word embeddings per comment
There is more than one way to create doc embeddings from a list of words but for this tutorial we used the average.
So this steps includes reading each preprocessed comment, getting the word embedding for each word (or unk), and then average them
into a final single vector for the document.  Here's the function for that:

```markdown
def create_doc_vectors(sentences,glove_vectors): 
    '''
    loop through tokens for each full comment
    get word vector for token or for unknown if token doesn't exist
    sum the word vectors and divide by number of tokens in comment at end

    params = nsents contains lists for all comments, each list contains tokens
    DV= list of doc vectors: the average for all token vectors
    length of DV will equal length of nsents = number of comments
    x is a np array to add all token vectors and store a running sum
    '''

    DV=[]                                  #list for doc vectors  
    
    for i,sent in enumerate(sentences):    #loop through each comment in nsents
        x = np.zeros((50))                 #np array to hold sum of word vectors for comment
        runsum = np.zeros((50))
        for word in sent:                  #loop through each token in comment
            if word not in glove_vectors:  #test if token in glove
                x=glove_vectors['unk']    #if not, get vector for unk and add it to x
            else:
                x=glove_vectors[word]     #else get vector for token and add it to x
                #print("word: ",word)
                #print(x)
            runsum+=x                     #add x to running sum of word vectors
        
        #print("sum x: ",runsum)
        #print("length x: ",len(sent))
        #print("ave for x: ",runsum/len(sent))
        DV.append(x/len(sent))             #divide x (the sum of word vectors) by the # of tokens = doc vector
        
    return DV
DV = create_doc_vectors(nsents,glove_vectors)
```
This required detailed testing, as it's not immediately obvious without doing the brute math that
your doc averages are in fact accurate averages of the word embeddings.  You can see all the print
statements commented out for that purpose.

### Step 5. Create doc embeddings as average from all word embeddings per Keyphrase
Here we create the doc embeddings for the keyphrases.
This required a little tokenization process for the keyphrases first:
```markdown
kpkeys = []
for tup in keyphrases:
    k,v = tup
    parts = k.split()
    kpkeys.append(parts)
print(kpkeys)

# output for that (note we generated 30 keyphrases in step 1 above)
[['newworld'], ['newworld', 'partner', 'network'], ['product'], ['times'], ['support'], ['difficult'], ['business'], ['customer'], ['office'], ['licensing'], ['halve', 'year'], ['triage', 'service'], ['issue'], ['advanced', 'ways'], ['information'], ['newworld', 'dynamics', 'gp'], ['client'], ['changes'], ['ms', 'blaze'], ['problems'], ['csp'], ['contact', 'person'], ['pain', 'points'], ['huge', 'market'], ['customer', 'needs'], ['great'], ['garage', 'company'], ['small', 'businesses'], ['partner', 'account', 'management', 'team'], ['months']]
```
Then we get into generating doc vectors for the keyphrases, alot like the doc vectors function above for customer comments

```markdown
# create doc vectors for keyphrases

KV=[]      #list to hold doc vectors
KeyV={}    

for k in kpkeys:                       #loop through list of keyphrases as lists of tokens
    #print("keyphrase: ",k)
    x = np.zeros((50))                 #np array to hold sum of word vectors for comment
    runsum = np.zeros((50))            #np array to hold running sum
    for keyword in k:                  #loop through each token in a keyphrase
        #print("keyword: ",keyword)
        if keyword not in glove_vectors:  #test if token in glove
            x=glove_vectors['unk']        #if not, get vector for unk and add it to x
            #print(x)
        else:
            x=glove_vectors[keyword]     #else get vector for token and add it to x
            #print("word: ",keyword)
            #print(x)
        runsum+=x                        #add x to running sum of word vectors

    #print("sum x: ",runsum)
    #print("length x: ",len(k))
    #print("ave for x: ",runsum/len(k))
    KV.append(x/len(k))             #divide x (the sum of word vectors) by the # of tokens = doc vector
 
#Finally we need to create a dict to map keyphrases so we can do our queries more naturally using keys
KeyV={}                  #dictionary of keyphrases as keys and doc vectors as values
i=0
for tup in keyphrases:   #loop through keyphrases as tuple list        
    k,v = tup            #get the keyphrase name as k, discard the value
    KeyV[k]=KV[i]        #store the name as the key and the doc vector from KV as the value, using index
    i+=1
    
# example indexing into this dictionary
# check out keyphrase vector
KeyV['product']
array([ 0.15882 , -0.27394 ,  0.25375 ,  0.76122 ,  0.30715 ,  0.71313 ,
       -0.59602 , -1.6259  ,  0.8165  ,  0.89072 ,  0.85715 ,  0.041891,
       -0.18236 , -0.55229 ,  0.69153 ,  0.43658 , -0.3366  ,  0.38019 ,
        0.40122 , -1.03    ,  0.6051  , -0.99571 , -0.068696, -0.012285,
       -0.38867 , -0.58734 , -0.62828 ,  0.2158  ,  0.66878 ,  0.19838 ,
        3.2414  , -0.051321,  0.091042, -0.35763 , -0.055053, -0.21982 ,
       -0.22069 ,  0.80755 ,  0.075926, -0.49719 ,  0.31928 , -0.30923 ,
       -0.14765 , -0.047711,  0.024934,  0.21341 ,  0.20546 ,  0.76339 ,
        0.3806  ,  0.70857 ])
```
### Step 6. Generate the Similarity Scores
Well here's the potential payoff of all that embedding work!
We have to build the similarity scores - and for this we use cosine similarity.
I have both custom code from our previous course and a quality check using scipy to verify the algo.
```markdown
# First the custom code
# functions for cosine similarity

def norm(vector):
    return sqrt(sum(x * x for x in vector))    

def cosine_similarity(vec_a, vec_b):
        norm_a = norm(vec_a)
        norm_b = norm(vec_b)
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return dot / (norm_a * norm_b)

# Similarity Test 1: using glove vectors
cosine_similarity(glove_vectors['man'],glove_vectors['peace'])
# output
0.3716446718756

# Similarity Test 2: using a keyphrase vector and Comment vector
#store single keyphrase vector to test similarity
X = KeyV['difficult']+KeyV['product']
# arbitrary sentence review to get index for Y
nsents[4]
# output
['licensing', 'and', 'no', 'support', 'for', 'older', 'technology', '.']
#store doc vector for comment
Y = DV[4]
# test similarity score
cosine_similarity(X,Y)
0.7902163297637051
# compare to scipy function for cosine similarity
1-distance.cosine(X,Y) 
0.7902163297637048
```
At this point we've confirmed out similarity code generates the same output as scipy - so that's good.
We'll have more work to do in the next step to see what similarity it telling us.

### Step 7. Generate queries and review similarity results - yea!
Here we go - time to test everything out on real data.
The way I approached this was to:
1. Generate similarity scores for all comments for each keyphrase -> creating a big grand list
2. Sort and slice off the top 10 comments with highest similarity to a keyphrase
3. Print out the top 10 comments for a select group of keyphrases

```markdown
# Here's the code for step 1
def get_top(keyphrase, simlist, sentences, keyphrase_dict):
    #get top 10 docs - based on similarity scores - for a given keyphrase
    '''
    params:
    key_index = index of keyphrase 
    simlist   = grandlist of sim scores: allcosim[]
    sentences = nsents - the original list of tokenized docs
    return:     res = list of results
    '''
    res = []
    #create a term to index dict for keyphrases
    t2i = {}
    for i,t in keyphrase_dict.items():
        t2i[t]=i
    
    allsimscores_for_akeyphrase = []
    
    #get all similarity scores for a given keyphrase
    for i, doc in enumerate(simlist[t2i[keyphrase]]):
        x = []
        x.append(i)
        x.append(doc)
        allsimscores_for_akeyphrase.append(x)
        
# Here's the code for step 2
    #sort and slice top 10
    t = sorted(allsimscores_for_akeyphrase,key=lambda x: x[1])
    t10 = t[:10]
    
    #print the top 10 results, and store in array
    topcosine_with_comments = []
    i = 1
    print('\n',"Top ten documents most similar to keyphrase: ",keyphrase,'\n')
    res.append(keyphrase.upper())
    for rank,cosinescore in t10:
        sentence = sentences[rank] 
        s = ' '.join(sentence)
        print(i,s)
        res.append(i)
        res.append(s)
        i+=1
    return res

# Here's the code for step 3
# generate some top 10 results for comments based on keyphrase
issue    = get_top('issue',allcosim, nsents, dict_keyphrases)
problems = get_top('problems', allcosim, nsents, dict_keyphrases)
changes = get_top('changes', allcosim, nsents, dict_keyphrases)
business = get_top('business', allcosim, nsents, dict_keyphrases)
months = get_top('months', allcosim, nsents, dict_keyphrases)
```
# Top 10 comments most similar to keyphrase: ISSUE
```markdown
1 newworld field has to be more in direct touch with partners , to really understand their needs , to know more about customer needs and be also more business oriented .   i still see the at subsidiary level msft field continues working in a different world vs partners and customers , sometimes even in a opposite way ( at least it seems to be like that ) . when us as partner hear from corp about overall strategies it motivates us to continue as partners , but when you try to land it at local subs it become a totally different story :-(
2 develop saas systems which run on linux but interact with   product x and viewport desktops so very little interaction with newworld
3 1 . the new ocp organization and partner engagement strategy is slow to be implemented . our company has been waiting for 8 months ( still counting ) before we are fully enrolled as a co  sell ready partner . internal processes related to p  seller enrollment are extremely slow , no one seems to have any control of it in the field . 2 . account executive are not responding despite the fact that ocp partner engagement managers do the go between 3 . the way the newworld org is supposed to work is not clear : ocp vs csu vs enterprise : who is really in charge of deployments ? 4 . ms services strategy has become very unclear . we used to partner deeply with newworld services . now , everything seems to be broken since the beginning of fy18
4 product x is too expensive . i would like to put our domain and all our office , communication and exchange stuff into the x cloud , but for a company with 11 employees in software development business all third party libs and services are too much . your premium devices are also   … are … pricy . :-}
5 easier access to sales materials . easier methods to purchase software and licenses . brochures , sales and marketing information need snail mailed out periodically . newworld branding needs emphasized . partners need to be able to get full working evaluation software to demonstrate the product to customers not crippleware
6 newworld has become totally distant with small resellers , etc . if i do n't hit the website , i am out of touch and i am not sure i am up to date . in addition , ms used to have tech and reseller meetings locally on at least a quarterly or half year basis and i gained a great deal of information at these events . finally i do n't think i have gotten an email from the reseller channel in several years now and have never found the place where i can review my communication with newworld
7 localisation , gdpr
8 we have little direct contact with newworld
9 nothing in the way of doing business but a painful point in that i had my newworld product stolen at a recent newworld tech event in birmingham nec , united kingdom . i did inform the nec , the police , the newworld tec event organization team  all to no avail . i have asked newworld if there is any way that they supply a replacement ( even a used one ) or a subsidized one but no one says anything ; which is pretty disappointing considering i have been a newworld professional for over 25 years and always help in the way of providing information ... etc   could you please see if something can be done , as you do n't expect to go to a ms tech event and have your tablet stolen ; i 'm lost without it .   thank you amir@ivitv.co 07866777744
10 when we want support , we need support ; first fling is to communities . not . my wants and needs are usually the same . paint points when we need support , we need support phone or chat or email . and , we need better marketing . i can help pdnrph@comcast.net
```
# Top ten documents most similar to keyphrase:  BUSINESS 
```markdown
1 no needs and no pain points :-)
2 little pain point , but to be expected as a result of new developments / changes in the cloud , the multiples of changes , documents and information   :-)
3 1 . the new xyz organization and partner engagement strategy is slow to be implemented . our company has been waiting for 8 months ( still counting ) before we are fully enrolled as a co  sell ready partner . internal processes related to p  seller enrollment are extremely slow , no one seems to have any control of it in the field . 2 . account executive are not responding despite the fact that xyz partner engagement managers do the go between 3 . the way the newworld org is supposed to work is not clear : xyz vs abc vs enterprise : who is really in charge of deployments ? 4 . newworld services strategy has become very unclear . we used to partner deeply with newworld services . now , everything seems to be broken since the beginning of fy18
4 develop saas systems which run on linux but interact with   product x and viewport desktops so very little interaction with newworld
5 product x is too expensive . i would like to put our domain and all our office , communication and exchange stuff into the x cloud , but for a company with 11 employees in software development business all third party libs and services are too much . your premium devices are also   … are … pricy . :-}
6 newworld field has to be more in direct touch with partners , to really understand their needs , to know more about customer needs and be also more business oriented .   i still see the at subsidiary level msft field continues working in a different world vs partners and customers , sometimes even in a opposite way ( at least it seems to be like that ) . when us as partner hear from corp about overall strategies it motivates us to continue as partners , but when you try to land it at local subs it become a totally different story :-(
7 years before we had a newworld employee as pam for direct interaction . this was very helpful . today we have sometimes pam from third party companies . no direct information . ist much harder to do business with newworld
8 when we want support , we need support ; first fling is to communities . not . my wants and needs are usually the same . paint points when we need support , we need support phone or chat or email . and , we need better marketing . i can help pdnrph@comcast.net
9 we have little direct contact with newworld
10 easier access to sales materials . easier methods to purchase software and licenses . brochures , sales and marketing information need snail mailed out periodically . newworld branding needs emphasized . partners need to be able to get full working evaluation software to demonstrate the product to customers not crippleware
```
### Final observations
A cursory review of top 10 comments based on different keyphrases shows a large overlap of the same comments.
While they show different positions in the list, and each list has on average about 3 unique items, overall we are not getting great differentiation.

### Next Steps
This was a great first exploration into the use of word and document embeddings as an application for searching customer comments for similarity.
While the comments varied considerably in length the keyphrases were mostly one word in length.
Perhaps results could be improved simply using more complex queries.
Also there are other approaches to keyphrase generation to be explored including phrase parsers like OpenIE.
Also we can explore using large pre-existing embeddings and also fine tuning those with our own customer corpora.
Finally we did not even look at using actual comments to search for similar comments - which naturally be more complex and in the language of the domain.
These are all fun directions to continue the exploration.
