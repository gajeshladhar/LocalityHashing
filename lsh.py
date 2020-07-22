# Author : Gajesh Ladhar
# WhatsApp : +918302019026
# LinkedIn :  https://www.linkedin.com/in/gajesh-ladhar-027a4870/


import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import twitter_samples

!wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

!gunzip '/content/GoogleNews-vectors-negative300.bin.gz'

embeddings=KeyedVectors.load_word2vec_format('/content/GoogleNews-vectors-negative300.bin',binary=True)

nltk.download('twitter_samples')
nltk.download("stopwords")

pos_tweets=twitter_samples.strings('positive_tweets.json')
neg_tweets=twitter_samples.strings('negative_tweets.json')

pos_tweets_orig=twitter_samples.strings('positive_tweets.json')
neg_tweets_orig=twitter_samples.strings('negative_tweets.json')

import re
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

tokenizer=TweetTokenizer(preserve_case=False)

for i,s in enumerate(pos_tweets):
  pos_tweets[i]=pos_tweets[i].lower()
  pos_tweets[i]=re.sub("@(.*?) ",' ',pos_tweets[i])
  pos_tweets[i]=re.sub("#",' ',pos_tweets[i])
  pos_tweets[i]=re.sub('http(s*)://(.*?)( |$)'," ",pos_tweets[i])
  pos_tweets[i]=re.sub('(-|!|([0-9]{1,}))'," ",pos_tweets[i])
  pos_tweets[i]=re.sub('(\.{1,})'," ",pos_tweets[i])
  pos_tweets[i]=re.sub('( ){1,}'," ",pos_tweets[i])
  pos_tweets[i]=tokenizer.tokenize(pos_tweets[i])
  
for i,s in enumerate(neg_tweets):
  neg_tweets[i]=neg_tweets[i].lower()
  neg_tweets[i]=re.sub("@(.*?) ",' ',neg_tweets[i])
  neg_tweets[i]=re.sub("#",' ',neg_tweets[i])
  neg_tweets[i]=re.sub('http(s*)://(.*?)( |$)'," ",neg_tweets[i])
  neg_tweets[i]=re.sub('(-|!|([0-9]{1,}))'," ",neg_tweets[i])
  neg_tweets[i]=re.sub('(\.{1,})'," ",neg_tweets[i])
  neg_tweets[i]=re.sub('( ){1,}'," ",neg_tweets[i])
  neg_tweets[i]=tokenizer.tokenize(neg_tweets[i])

stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

for i,s_list in enumerate(pos_tweets):
  for j,s in enumerate(s_list):
    #print(str(i)+" "+str(j))
    pos_tweets[i][j]=stemmer.stem(s)

for i,s_list in enumerate(neg_tweets):
  for j,s in enumerate(s_list):
    neg_tweets[i][j]=stemmer.stem(s)

for s_list in pos_tweets:
  for s in s_list:
    if s in stopwords_english :
      s_list.remove(s)


for s_list in neg_tweets:
  for s in s_list:
    if s in stopwords_english :
      s_list.remove(s)

tweets=pos_tweets+neg_tweets

embeddings_tweets={}
for i,tweet in enumerate(tweets):
  doc_vec=np.zeros(shape=(300,1))
  for word in tweet :
    try:
      doc_vec+=np.array(embeddings[word]).reshape((300,1))
      
    except :
      continue
  embeddings_tweets[i]=doc_vec
  if(i%500==0):
    print("Completed till now : "+str(i))

np.sum(embeddings_tweets[0]-embeddings_tweets[10])

# Locally Sensitive Hashing...
N_Planes=15
N_Universe=10

planes=np.random.normal(size=(N_Universe,300,N_Planes))
"""for i in range(N_Universe):
  for j in range(N_Planes):
    idx=np.random.randint(0,10000)
    planes[i,:,j]=embeddings_tweets[idx].reshape((300))
"""
# single universe
# single vector
def get_hash(v,plane):
  t_hash=np.dot(v.T,plane).reshape((N_Planes))
  hash=0
  for i,t in enumerate(t_hash):
    if t>=0 :
      hash+=2**i
  return hash

# Making Hash Tables
Hash_Tables=[]
for universe in range(N_Universe) :
  hash_table={i:[] for i in range(2**N_Planes)}

  for i in list(embeddings_tweets.keys()):
    hash=get_hash(embeddings_tweets[i].reshape((300,1)),planes[universe])
    hash_table[hash].append(i)
  Hash_Tables.append(hash_table)

Hash_Tables[0]

# select k neibours
def k_neibours(vectors,pred,k=1):
  nearest=[]
  for vec in vectors:
    nearest.append((np.dot(vec.T,pred)*1.0/(np.linalg.norm(vec)*np.linalg.norm(pred))).reshape((1))[0])
  nearest=np.argsort(nearest)
  return nearest[-k:]

# Find Relative Tweet 
def find_relative(tweet_id):
  pred=embeddings_tweets[tweet_id]
  
  probable_sets=[]
  for i,table in enumerate(Hash_Tables):
    hash=get_hash(pred,planes[i])
    for probable in table[hash]:
      probable_sets.append(probable)

  probable_sets=list(set(probable_sets))

  if tweet_id in probable_sets:
    probable_sets.remove(tweet_id)
  
  probable_vectors=np.zeros((len(probable_sets),300,1))

  for i,val in enumerate(probable_sets):
    probable_vectors[i]=embeddings_tweets[val]

  print("Searchable Values : "+str(len(probable_sets))+"\n")
  indexes=k_neibours(probable_vectors,pred,k=2)
  temp=[]
  for i in indexes :
    temp.append(probable_sets[i])
  indexes=temp
  return indexes

index=np.random.randint(0,10000)
pred_i=find_relative(index)
print("User : "+ (pos_tweets_orig+neg_tweets_orig)[index])
print("")
print("ChatBot 1 : "+((pos_tweets_orig+neg_tweets_orig)[pred_i[0]]))
print("ChatBot 2 : "+((pos_tweets_orig+neg_tweets_orig)[pred_i[1]]))

