# GenerativeAI
Learning LLM with Generative AI

NLP Evolution

-------------------------------
-------------------------------

1) Rule Based System Era.
  a) Syntax analysis --> Techniques such as parsing and part of speech tagging for understanding sentence structure .
   ==> Parsing is the process of analysing senetences to determine their structure and the relationshipss between the words  . This technique breaks down sentences into their consttuent part revelaing the underlying hierarchical organisation .

Drawbacks :

Complexxity of NLP, Language Ambiguity words and things meaning differ based on context , Scalability

---------------------------------------
---------------------------------------

2) Statistical NLP Era

 a) Data Driven Approaches

 b) Probabilty and Statistics

 c) Language Amibiguity( Adapting to new Language Patterns) : Using n-grams and probabilistic language models to predict word sequences .

  Language Models : predict the likelihood of a word or a sequence of a word in giving context improving the NLP systems understanding of language structure .

  N-grams are contiguous sequence of n words used to estimat the probablity of a word sequence in large corpus of text .

2 grams(bigrams): "The cat""cat sat" "sat on ""on the ""the mat"
3 grams(trigrams): "The cat sat" "cat sat on" "sat on the" "on the mat"

Hidden Markov Models : Tackling NLP tasks with probabilties .

(Part of Speech-tagging & Named Entity Recognition -> Alice(NAME) visited(LOCATION) last weekend(DATE))
Example : HMM use probabilties to detrmine the most likely sequence of grammatical tags for words in a sentence , making it easier to anlyse senetence structure .

Limitations : 

Data Sparsity is a challenge as many word combinations are rarely observed making it difficult for statistical model to accruately estimate probailties


Lack of Semantic understanding

---------------------------------
-----------------------------------

3) Maching Learnning Era :

1)Algo such as Naive Bayes , Support Vector Machines , and neural networks were introduced to tackle a wide range of NLP tasks including text classification , sentiment analysis and machine learning translation .

2) ML approaches allowed NLP systems to handle larger scale data , further enhancing their language processing capabilities .

ALgo's 

Uses of Naive Bayes and Support Vector Machines(linear models) for text classification . They laid the groundowrk for more complex techniques like Neural Networks .

Neural Networks :

Neural Networks are adaptable and capable of continuous learning allowing them to better handle changing language patterns and new data .

Automatic feature learning : Neural Networks automatic learn meaningful features abd represntations from raw text data , eleiminating the need for manual feature engineering .

RNN( Recurrent Neural Networks : RNN are a specific kind of neural networks tailored for handling sequential data which makes them well suited for NLP tasks. They process input sequences step by step using a hidden state to retain information from earlier elements allowing them to comprehend relationships between words in a sentence.

Tasks for RNN :

Machine Translation : Translating text from one language to another .
Text Summarization : generating concise summaries of longer text passages .
Sentiment Analysis : Classifying the sentiment of a piece of text .


Limiattions of RNN : Difficulry in modeling the long term dependencies a new architecture called long short-term memory or LSTM was introduced.

LSTM are a type of RNN architecture that incorporates special memory cells enabling them to retain information over longer sequences and effectively learn long range depedencies .

Example of LSTM TASKS :

1) Language Modelling : LSTM can predict the next word in a sentence , considering not only recent words but also those from earlier parts of the text .

2) Text Generation : LSTM can generate coherent and contextually relevant text by learing long -range patterns in the training data .

3) Machine translation : LSTM can better handle long sentences and complex structures in source and target languages resulting in more accurate translations .


--------------------------------------------------------------
------------------------------------------------------------------

4) Embeddings Era 

The Embeddings Era signaled a significant transformation in NLP with reseracjers starting to represent words as continuous vectors within a high -dimensional space . This allowed models to better capture the semantic and syntactic connections between words compared to earlier techniques .

Techniques used :

1) WORD2VEC AND GLOVE :  Continuous vector represntation of words.

High - dimensional space : Representing words as continuous vectors in high dimensional space allows models to capture semantic and syntactic relationships more effectively .

Unsupervised techniques : Popular techniques like Word2Vec and GloVe use unsupervised methods to generate meaningful numerical representations of words .

2)  Contextualised word embeddings :  Capturing Context-Dependent  Word meanings .

ELMO : ELMo introduced contextualised word embeddings that capture word meanings based on their specific context resulting in more accurate representations .


Limitations :

1) Fixed Length Reprentations : limit the ability to capture the full complexity of language as they cannot easily accomodate variations in context or meaning . This constraint can hinder the model's performance in understanding and processing naunced linguistic structures .

2) Lack of Transfer Learning :  The absence of transfering learning implies that models often need to be trained from scratch for each new task which can be time - consuming and computationally expensive . This limitation prevents the efficient reuse of pre-existing knowledge slowing down the developent and deployment of NLP models . 


This limitations were covered by transformers .
