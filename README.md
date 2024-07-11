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

The Embeddings Era signaled a significant transformation in NLP with researchers starting to represent words as continuous vectors within a high -dimensional space . This allowed models to better capture the semantic and syntactic connections between words compared to earlier techniques .

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

---------------------------- 
-----------------------------

Transformers


This allows models to capture dependencies between words and cncpet efficiently regardless of distance between them 

They also enabled parallel computation making them faster or at least more hardware -efficient than RNNs and LSTMs during training particularly on larger datasets .

Transformer architecrure : 

1) There are mainly two key concept of Encoder -Decoder .

Encoder is responsible for processing the input text while decoder generates the output text . To illustrates this , consider a machine translation task where the enocoder reads a sentence in one language and the decoder translates it into another language .

Example :

"Hello world"-> encoder -> knowledge -> decoder -> "Hola mundo"

2) Attention mechanism 

The attention mechanism in neural networks helps capture  relationship between elements in a sequence by assigning different importance weights to each element . It allows the model to focus on most relevant parts of the input improving its ability to understand context and handle long -range dependencies .

Example : This is similar to human beings when we are going through a sentence , we don't use focus on individual words we also pay attention to how they relate to each other and how they affects the overall meaning .

Limitation : Because of this it loses concept of position and word order .

Attention mechanism of tranformer consists of several componenets :

a) Self-Attention : Enables the model to weigh the importance of word -pairs relative to one another and understand relationships between words .

For example:

"The cat chased the mouse , and it scurried away"

The self attention mechanism help the model to figure out that it refers to the mouse.In turn self -attention mechanism computes attention scores for each word pairs and uses these scores in order to generate context -aware reprentations of the input .

b) Scaled Dot-Product attention: Computes attention scores between words in a sequence through (dot products, scaling and softmax.These process genrates attention scores) . which tell model how much focus should be placed on specific word when considering it in the sequence

Example : 

"The cat sat on the mat"

Q_sat*K_the,Q_sat*K_cat,Q_sat*K_on,Q_sat*
K_the_2,Q_sat*K_mat

Scale down ny sqrt(d)

Apply softmax

Compute weighted sum of Value vectors

Obtain context-ware representation of "sat"

c) Multi-Head attention : Employs multiple "heads" to simultaneously capture different aspects of input data such as suntactic and semantic relationships for a more comprehensive understanding.

For example :

Sentence : " The cat chased the mouse , and it scurried away".

Head1(Syntactic relationships):
- "cat" and "chased"
-"mouse" and "scurried"
- "it" and "mouse"

Head2(Semantic relationships):
-"chased" and "scurried"
-"cat" and "mouse"

Like this by combining output of all the heads and having access to different perspectives from multiple specilaized components , the model can get more comprehensive understanding of the input .


3) Positional Encoding(Keepings words in Order) : Used in Transformer models to provide information about the position of words in a sequence as the attention mechanism does not inherently capture word order . It involves adding unique , learnable vectors to the input embeddings , allowing the model to recognise the order of words and understand the structure of sentence .

==> Positional Encoding is added to the input embeddings before they are fed into the model ensuring that model can recognize the orders of words inside the input .

This technique is important for transoformer to understand the structure and meaning of sentences . Without it they will basically see a messy bag of words.


Sentence : "When life gives you lemons."

Tokens: ["When","life","gives","you","lemons"]

Embeddings:[e_When,e_life,e_gives,e_you,e_lemons]

Positional Encodings:[p_1,p_2,p_3,p_4,p_5]

Combined Input:[e_When+p_1.e_life+p_2,e_gives+p_3,e_you+p_4,e_lemons+p_5]

Postional Encoding specifically designed for transformers as in RNNs and LSTM automatically capture word order through their inherent sequential processing though that naturally makes them much slower.

4) Feed_forward Networks (Making sense of words)

a) Position-WISE FEED-FORWARDED NETWORKS

Inside a Transformer model , feed-forward networks(FFNs)
are used in both the encoder and decoders layers to learn complex, non-linear relationships between the input embeddings and their context . These FFNs consist of fully connected layers that process the output of the attemtion mechanism , further refining the model understanding of the input data . 

After attemtion mechanism has done its job of capturing the relationships between words, the output is passed through a feed-forwarded network .

It is common component in many other neural networks.They operate independly of each position in th einput sequence which allows for parallel computation and contributes to relative compute efficiency of the transformer .

5) Layer Normalisation ( Stabilising Training)

Inside a transformer model, Layer Normalisation is a technique that normalises the activations in a layer by scaling and centering them , resulting in improved training stability and faster convergence . Applied to both the enocder and decoder layers , it helps mitiagate the vanishing gradient problem( to have consistent mean and variance. This basically happens when floating point numbers in computers get way too close or equally as bad way too far from zero and they become hard to accurately work with and represent ) and facilitate deeper model architectures .

Note : It is also applied after both the attention mechanism and feed-forward network within each layer of the model .

This technique is also used in other deep learning networks such as Convolutional neural networks and RNNs.

Now will learn in more detail the encoder and decoder .

==> They work together to process and to generate natural language making it effective for various NLP tasks.

==> Identical layer stacks : Bothe encoder and decoder built using stacks of identical layers which conatins multi-head self-attention, position-wise feed -forward networks, layer normalisation and residual connections.

The Encoder processes the input text into a continuous representation that captures word relationships and context, with each layer refining the representation to learn complex patterns and dependencies such as generating context-rich reprenetation for machine translation .

Example :

Sentence: "The cat sat on the mat."

Pass Sentence through encoder:

Sentence -> Encoder -> [ 0.14,-1.23....7.90](continuous vector reprentation)


This representation can be used by the decoder or othe parts of the model to perform various NLP tasks such as machine translation , summarisation or question -answering.

The Decoder : It generates the output text using the encoder's continuous representation, employing self-attention and encoder-decoder attention to focus on relavant input parts such as generating answers in a question -answering system based on the encoder -processed question and context.

It also employs self attention focusing on different parts of the input representation .

Example :

Sentence: "The cat sat on the mat."

Encoder representation [0.14,-1.23....7.90]

Pass through the decoder:
[0.14,-1.23....7.90] -> Decoder -> "EI gato sento en la alformbra.


Example : Question -Answering System


Passage "During the summer, Raju enjoys going to beach with his friends. In the winter , he prefers staying indorrs, reading books and drinking hot chocolate .


Question: "Does Raju like cold weather?"

Pass through encoder:

Passage + Question -> Encoder ->[1.44.0.65....-4.36](continuous vector representation)

(Here Enocder processes the input question and context  , undersatnding that during summer he enjoys on beach and above in winters ). From Passage and Question it creates joint vector representation that is then passed to decoder)

Pass through Decoder:

[1.44.0.65....-4.36] -> Decoder -> Abstractive answer .

Abstractive answer : "Raju seems to dislike cold weather as she prefers stay indorrs during winter"
