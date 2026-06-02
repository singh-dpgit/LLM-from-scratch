with open("Verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
print("Total no. of characters-->",len(raw_text))
print(raw_text[:250])
import re #library for Regular expression operations module
preprocessed=re.split(r'([.,:;?_!"()\']|--|\s)',raw_text)#preprocessed=re.split(r'(\s)',raw_text)#split the whole text whenever white(\s) space is encountered
preprocessed=[item  for item in preprocessed if item.strip()]#removing white spaces to save memory & computing requirement ,this two lines makes the tokenizer.
print(preprocessed)
"""
#tokenization done--->next step create token id's
Each unique token is mapped to an unique integer called token Id
"""
all_words=sorted(set(preprocessed))#create a set so that no word repeats and sort it to assign unique ids easily
vocab_size=len(all_words)#no repeatation of words hence size always less
print(vocab_size)
vocab={token:integer for integer,token in enumerate(all_words)}
"""
enumerate-->takes every word/token and assigns a unique integer value to every token
the above line says-->assign integer to every token in 'all_words'
vocab-->a dictionary key value pair(token:integer)
**ENCODER**-->TAKES INPUT AS TOKENS AND RETURNS TOKEN ID'S
"""
for i,item in enumerate(vocab.items()):#vocab.items()return(key,value)pairs coverts dictionary pairs to individual items
    print(item)#print tokens and their token ids till 50
    if i>=50:
        break
"""we are not printing i which is index assigned to every item
enumerate assigns index just as before it did here it is used to just check the
condition to print only first 51 items otherwise it will print every item
now**DECODER**-->TAKES input as token ID's and return tokens which
further gets converted into sentences as our output.
SO TO IMPLEMENT BOTH DECODER AND ENCODER WE WILL MAKE A CLASS
"""
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int=vocab#str_to_int became a ditionary of token and their corresponding token ids
        self.int_to_str={i:s for s,i in vocab.items()}#for decoder to work by reversing the key value pair of vocab we get int_to_str dictionary which is used in decoder to convert token ids to tokens
    def encode(self,text):
        preprocessed=re.split(r'([.,:;?_!"()\']|--|\s)',text)
        preprocessed=[item  for item in preprocessed if item.strip()]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        #Replace spaces before the specified punctuation
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text
tokenizer=SimpleTokenizerV1(vocab)
text="""ANOTHER MAN stands next to him. 
The Second Man reaches behind the First Man's 
back and puts a discreetly folded ten-dollar bill into his hands.
These Two Men from the front.  Both somber, in their early fifties.  
They begin to walk down the aisle of the funeral parlor."""
ids=tokenizer.encode(text)#ids is a list containing all the token ids
print(ids)
"""sentence must be from vocab otherwise error if any word not found keyerror
this is the reason why training dataset must be huge so that we dont
face error"""
text=tokenizer.decode(ids)#text--> string built from token ids provided in Ids(list)
print(text)
"""To handle unknown words GPT uses a technique
--> Adding special context tokens
This will handle unknown words
for this we will create version2-->SimpleTokenizerV2
add 2 more token <|unk|>(unknown word) & <|endoftext|>
When working with multiple text sources,we add|<endoftext>|tokens between these
texts.These|<endoftext>|tokens act as markers,signalling the start
or end of a particular segment.benifit in processing and understanding different
text efficiently..
"""
all_tokens=sorted(list(set(preprocessed)))#same command as earlier adding two more tokens in preprocessed
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab={token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))#2 extra since added
for i,item in enumerate(list(vocab.items())[-5:]):#checking last 5 entries
    print(item)
class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}#for decoder to work
    def encode(self,text):
        preprocessed=re.split(r'([.,:;?_!"()\']|--|\s)',text)
        preprocessed=[item  for item in preprocessed if item.strip()]
        preprocessed=[item if item in self.str_to_int#EXTRA PART ADDED
                      else "<|unk|>" for item in preprocessed#EXTRA PART ADDED
        ]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        #Replace spaces before the specified punctuation
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text
tokenizer=SimpleTokenizerV2(vocab)
text1="Hello budddy! How are you?"
text2="In the sunlit terraces of the palace."
text="<|endoftext|>".join((text1,text2))#join takes one argument here text1 and text2 are rapped in one tupple
print(text)
print(tokenizer.encode(text))#prints <|unk|> ids wherever token is unknown word
print(tokenizer.decode(tokenizer.encode(text)))#print original text but prints <|unk|> in place of unknown tokens
"""
similarly different token can be used to handle many situations
[BOS](begining of sequnce):marks the start of a text,signifies where a piece of contain begins
[EOS](end of sequence):at end of text,similar use as <|endoftext|>
[PAD](padding):when training LLMs with batch sizes larger than one,
the batch might contain texts of varying lengths.To ensure all texts have the
same length,the shorter texts are extended or'padded' using the [PAD] token,
up to the length of the longest text in the batch.
"NOTE:"GPT doesnt uses any of the above mentioned tokens 
but only use <|endoftext|> token for simplicity.
"NOTE:"GPT doesnt use <|unk|>token for unknown tokens
it uses 'BYTE PAIR ENCODING'that breaks down words into subward units.
"""
"""---------BYTE PAIR ENCODING---------------------"""
import tiktoken
from importlib_metadata import version
print("tiktoken version:", version("tiktoken"))#print version of tiktoken
tokenizer=tiktoken.get_encoding("gpt2")#uses similar to SimpleTokeniozer we used before 
"""tiktoken library ka ek function hai get_encoding jo input le rha
version of gpt according to which tokenizer kaam krega aure ye function
return krega ek object jisko save kiya gaya hai tokenizer mein"""
text=(
      "Hello, Buddy how are you doing? <|endoftext|>In the sunlit terraces" \
      " of some unknown place letsroamsomewhere!"
)#since BPE subword are formed hence 'letsroamsomewhere' no out of vocab problem
integers=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(integers)#prin token id's
string= tokenizer.decode(integers)
print(string)
"""<|endoftext|> is assignes the last token id 
so token id of <|endoftext|> can be said as size of token ids
in gpt2 its 50256 whereas english vocab consists of approx
170k-200k words so no. of token ids are also reduced by BPE/subward
token scheme 
--------------one more example -------------"""
integers=tokenizer.encode("Arkwbstd om")#though this word not exist still no error
print(integers)
string=tokenizer.decode(integers)
print(string)
"""----------------------Input-targetpairs-------------------------"""
"""In this section we implement a data loader that fetches the input-target
pairs using a sliding windown approach"""
with open("Verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
enc_text=tokenizer.encode(raw_text)
print(len(enc_text))
#now we remove 1st 50 tokens for dataset for demonstration
enc_sample=enc_text[:50]
print(len(enc_sample))
#creating mechanism of input-target pairs
#context_size determines how many tokens are included in the input
context_size=4#length of the input
"""context_size of 4 means the model is trained to look at a sequence
of upto 4 words to predict the next word in sequence.The input'x'is the first
4 tokens[1,2,3,4] and the target 'y'is the next 4 tokens[2,3,4,5]"""
x=enc_sample[:context_size]
y=enc_sample[1:context_size+1]
print(f"x: {x}")#input
print(f"y:       {y}")#output
for i in range(1,context_size+1):
    context=enc_sample[:i]#input
    desired=enc_sample[i]#target LLM is supposed to predict
    print(context,"-------->", desired)
#NOTE:NO. OF PREDICTION TASKS = CONTEXT_SIZE
for i in range(1,context_size+1):#id's to text form
    context=enc_sample[:i]#input
    desired=enc_sample[i]#ouput
    print(tokenizer.decode(context),"----->", tokenizer.decode([desired]))
#desired is a single integer but tokenizer.decode() takes list hence made into list
"""----Implementing an efficient dataloader that iterates
over the input dataset and returns and targets as
PyTorch,which can be thought of as multidimensional arrays.
In particular,we are interested in returning two tensors:
an input tensor containing the text that the LLM sees and a target tensor
that the LLM predicts """
from torch.utils.data import Dataset,DataLoader
class GPTDatasetV1(Dataset):#max_lenth=contextsize,txt=Verdict/data,tokenizer=BPE
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]
    #Tokenize the entire text
        token_ids=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
    #Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0,len(token_ids)-max_length,stride):#stride determines how much we slide if stride=1-->highly overlapping sequences,stride=4-->no overlapping sequences
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):#returns the row of the particular index given(for dataloader)
        return self.input_ids[idx], self.target_ids[idx] #if index is 50 the input will be 50th row of input tensor and output will be 50th row of output tensor
"""------------IMPLEMENT DATALOADER-----------------"""
def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):#batch_size=no. of cpu cores (by default=4),num_workers=no. of cpu threads
    #GPT learns next-token prediction.
      #If you always feed sequences in text order, the model may see long chains of highly correlated sequences in the same batch.
      #This can slow learning or bias gradients.
      #Shuffling ensures batches are randomly sampled, which improves generalization.
      # GPT models expect fixed-size tensors per batch: [batch_size, max_length].If the last batch has fewer samples, 
      # your input batch would have a smaller batch dimension, which could break your training loop if you’re using:input_batch.to(device)
      #  or some matrix operations that assume [batch_size, max_length].Solution: drop_last=True ensures all batches are uniform.
      # num_workers=4	four workers → can fetch multiple batches simultaneously.
      # Especially useful for large datasets like full books (millions of tokens).
      # Each worker handles subset of indices, DataLoader merges them into batches.
        #Initisalize the tokenizer
    tokenizer=tiktoken.get_encoding("gpt2")
    #create dataset
    dataset= GPTDatasetV1(txt,tokenizer,max_length,stride)
    #create dataloader
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)#dataloader is going to look at the getitem() in dataset and going to return the input target pairs
    return dataloader
with open("Verdict.txt","r",encoding='utf-8') as f:
    raw_text=f.read()
    #Convert dataloader into a Python iterator to fetch the next entry via Python's built-in next() function
import torch
print("Pytorch version:",torch.__version__)
dataloader=create_dataloader_v1(raw_text,batch_size=1,max_length=4,stride=1,shuffle=False)
data_iter=iter(dataloader)
first_batch=next(data_iter)
second_batch=next(data_iter)
print(first_batch)
print(second_batch)
#NOTE:That an input size of 4 is relatively small and only choosen for illustration purpose.
#It is common to train LLM's with input sizes of atleast 256
#Effect of batch size
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=4,stride=4,shuffle=False)
data_iter=iter(dataloader)
inputs,targets=next(data_iter)
print("Inputs:\n",inputs)
print("Targets:\n",targets)
"""--------------VECTOR EMBEDDINGS--------------------"""
import gensim.downloader as api
model=api.load("word2vec-google-news-300")#using already pretrained data
word_vectors=model
print(word_vectors['computer'])
"""--------------BUILDING VECTOR EMBEDDING------------"""
input_ids=torch.tensor([2,3,5,1])
vocab_size=6#for sake of simplicity we take dim=3 and vocab=6
output_dim=3
torch.manual_seed(123)#on every run we will get same random values in embedding layer since we are setting the seed so that at once random values are generated and these values are optimised during training of LLM as a part of LLM optimization itself
embedding_layer=torch.nn.Embedding(vocab_size,output_dim)#Embedding creates a dictionary(look-up table) and initialise all the weights of the embeddings matrix randomly 
print(embedding_layer.weight)#weight is a 2d matrix of size vocab_size*output_dim
"""This gives the weight matrix
(6 rows-->bcoz 6 vocabs and 3 colmns-->since 3dimensions for each vocab)
 of embedding layer which contains
small,random values and these values are optimized during LLM training
as a part of the LLM optimization itself (in upcoming topics)"""
#look-up table--> bcoz we can look at the row of the particular id of vocab and get the row/vector corresponding to that vocab
for i in range(0,6):
    print(embedding_layer(torch.tensor([i])))#ith row(0-5(all row line by line)) of our matrix before look-up operation
print(embedding_layer(input_ids))#for token ids 2=3(since starts form 0),3->4,5->6,1->2 as assigned before
"""----------------------POSITIONAL EMBEDDINGS-------------------------"""
vocab_size=50257
output_dim=256
token_embedding_layer=torch.nn.Embedding(vocab_size,output_dim)#returns a tensor object which has special methods like __call__() which is used to get the row of the particular token id from the embedding matrix
max_length=4
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False)
data_iter=iter(dataloader)#dataloader helps us to manage the task of inputing,batching,creating different batches,parallel processing becomes more easier
inputs,targets=next(data_iter)
print("Token ID's:\n",inputs)
print("\nInputs shape:\n",inputs.shape)
#8*4 dimensional token id tensor since batch_size=8 and max_length i.e,context_size=4
#for every input id in 8*4 tensor lets get 1 row(256 dimensions/cols) that is vector embedding layer(look-up table)
token_embeddings=token_embedding_layer(inputs)#implicitly calls the __call__() method of the embedding layer which returns the row of the particular token id from the embedding matrix
print(token_embeddings.shape)
context_length=max_length
pos_embedding_layer=torch.nn.Embedding(context_length,output_dim)
pos_embeddings=pos_embedding_layer(torch.arange(max_length))#create token ids from 0 to max_length-1 and assign positional embeddings
print(pos_embeddings.shape)
import torch

vocab = {
    "Your": 0,
    "journey": 1,
    "starts": 2,
    "with": 3,
    "one": 4,
    "step": 5
}
sentence = ["Your", "journey", "starts", "with", "one", "step"]
input_ids = torch.tensor([vocab[word] for word in sentence])
print(input_ids)
torch.manual_seed(123)
embedding = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
inputs = embedding(input_ids)
print(inputs)
"""-------------------Implementing simplified self attention mechanism----------------"""
import torch
inputs = torch.tensor([
[0.43, 0.15, 0.89], # your
[0.55, 0.87, 0.66], # journey -->embedduing of the word journey which is a vector of 3 dimensions and highly realted to word starts than other words therefore it will wrt to joruney the attention score of starts will be higher than other words
[0.57, 0.85, 0.64], # starts
[0.22, 0.58, 0.33], # with
[0.77, 0.25, 0.10], # one
[0.05, 0.80, 0.55]  # step
])
query=inputs[1]
attn_scores_2=torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i]=torch.dot(x_i,query) #Dot product of the query vector with each input embedding vector to get attention scores abcosθ suppose a and b are the two vectors of words starts and journey and we want to calculate which word is more related to journey among all the words in the sentences therefore our query becomes the embedding of the word journey and we take dot product of this query with all the input embeddings to get attention scores which will be higher for the word starts than other words since it is more related to journey than other words in the sentence 
print(attn_scores_2)#printing attention scores of query w.r.t every input embedding
"""
why dot product?-->since it gives us a measure of similarity between two vectors,
if vectors of two words are closer in the vector space, their dot product will be higher,indicating a stronger relationship between those words.
assume they are parallel then the angle between them is 0 degree and cos0=1 hence dot product is maximum
if they are orthogonal then the angle between them is 90 degree and cos90=0 hence dot product is 0
if they are opposite then the angle between them is 180 degree and cos180=-1 hence dot product is minimum
This is how the attention mechanism helps the model to focus on relevant parts of the input when making predictions,by assigning higher attention scores to more relevant words in the context of the query word.
"""
"""
now we will normalise the attention scores(in terms of %age so that they sum up to 1) using softmax function which is given by the formula
softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j=1 to n
where x_i is the attention score for the i-th input and n is the total number of inputs. This function converts the attention scores into probabilities,
 allowing the model to weigh the importance of each input when making predictions.
 The higher the attention score, the more relevant the input is to the query, and thus it will have a higher probability after applying softmax. 
 This helps the model to focus on the most relevant parts of the input when generating output.
"""
attn_weights_2_tmp=attn_scores_2/attn_scores_2.sum() # very basic normalisation technique where we divide each attention score by the sum of all attention scores to get the attention weights which sum up to 1
print("Attention weights:",attn_weights_2_tmp)#attention scores and attention weights are same only difference is that attention weights are normalised attention scores and sumup to 1 whereas attention scores are not normalised and can have any value
print("Sum of attention weights:",attn_weights_2_tmp.sum())
"""<----softmax function for normalisation----> expression = exp(x_i) / sum(exp(x_j)) for j=1 to n"""
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weight_2_naive=softmax_naive(attn_scores_2)
print("Attention weights: ",attn_weight_2_naive)
print("Sum of attention weights=: ",attn_weight_2_naive.sum())
"""
Drawback of the above implementation is that it can lead to numerical instability when the attention scores are large,
 as the exponential function can produce very large values, leading to overflow.
"""
#Implmenting softmax using PyTorch's built-in function(advisable to use it since it is optimized for numerical stability and performance)
attn_weights_2=torch.softmax(attn_scores_2,dim=0)
print("Attention weights: ",attn_weights_2)
print("Sum of attention weights: ",attn_weights_2.sum())
"""
now make the final context vector by multiplying each input embedding vector by its corresponding attention 
weight and summing the results to get a single context vector that represents the weighted 
average of the input embeddings based on their relevance to the query. This context vector can then be used by the model to make predictions or generate output,
as it captures the most relevant information from the input based on the attention mechanism.
"""
query=inputs[1]
context_vector_2=torch.zeros(inputs.shape[1])#initialise context vector with zeros of the same dimension as input embeddings
for i,x_i in enumerate(inputs):
    context_vector_2+=attn_weights_2[i]*x_i #multiply each input embedding vector by its corresponding attention weight and sum the results to get the context vector which is a weighted average of the input embeddings based on their relevance to the query
print("Context vector:",context_vector_2)   
"""
till now we found context vector of only second word (journey)
now we will find context vector for all the words in the sentence by treating each word as a query and calculating its 
attention scores, attention weights, and context vector in the same way as we did for the second word (journey).
 This will give us a context vector for each word in the sentence, 
which can be used by the model to make predictions or generate output based on the relevance of each word to the others in the sentence.
"""
attn_scores=torch.empty(6,6)
for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores[i,j]=torch.dot(x_i,x_j)
print("Attention scores:\n",attn_scores)
#for loop is not effcient way to calculate attention scores since it has time complexity of o(n^2)
#we can calculate attention scores more efficiently using matrix multiplication(matrix(M) * transposeof(M))
attn_scores_efficient=inputs @ inputs.T #matrix multiplication of input embeddings with its transpose to get attention scores more efficiently
print("Attention scores (efficient):\n",attn_scores_efficient)
#let's normalise the attention scores using softmax function to get attention weights
attn_weights=torch.softmax(attn_scores,dim=-1)
print("Attention weights:\n",attn_weights)
#now we will calculate the context vectors for all the words in the sentence by multiplying the attention weights with the input embeddings and summing the results
print("All row sum:\n",attn_weights.sum(dim=-1))#to check if all the rows sum up to 1 since we applied softmax function to normalise the attention scores)
#Therefore now the final step
all_context_vectors=attn_weights @ inputs #matrix multiplication of attention weights with input embeddings to get context vectors for all the words in the sentence
print("Context vectors for all words:\n",all_context_vectors)
"""
Though the above implementation of self attention mechanism is good
but it only pays attention to the words which are more semantically related to query
but it does not pay attention to the position of the words in the sentence which is also important for understanding the meaning of the sentence.
For example, in the sentence "The cat sat on the mat because it was warm",
here the words "mat" and "warm" might not be closer in traditional vector space but they are related in the context of the sentence
 since "mat" is warm and "warm" is a property of "mat".Hence the concept of "Trainable Weights" is introduced int the attention mechanism.
"""
"""<-----------------------------Self attention mechanism with trainable weights(Query,Key,Value)------------------------------>"""
import torch
inputs=torch.tensor(
[
[0.43, 0.15, 0.89], # your
[0.55, 0.87, 0.66], # journey
[0.57, 0.85, 0.64], # starts
[0.22, 0.58, 0.33], # with
[0.77, 0.25, 0.10], # one
[0.05, 0.80, 0.55]  # step
]
)
d_in=inputs.shape[1]#dimension of input embeddings(vector embeddings) which is 3 in this case
d_out=2
torch.manual_seed(123)
W_query=torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#generating random weights for query matrix of size d_in*d_out and setting requires_grad to False since we are not going to train these weights in this example
W_key=torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#generating random weights for key matrix of size d_in*d_out and setting requires_grad to False since we are not going to train these weights in this example
W_value=torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#generating random weights for value matrix of size d_in*d_out and setting requires_grad to False since we are not going to train these weights in this example
print(W_query)
print(W_key)
print(W_value)
x_2=inputs[1]#query vector for the word journey
query_2=x_2 @ W_query #matrix multiplication of query vector with query weights to get the query representation for the word journey
key_2=x_2 @ W_key #matrix multiplication of query vector with key weights to get the key representation for the word journey
value_2=x_2 @ W_value #matrix multiplication of query vector with value weights to
print("Query representation for the word 'journey':",query_2)
#now lets do it for the whole input vectors matrix
keys=inputs@W_key#keymatrix
values=inputs@W_value#value matrix
queries=inputs@W_query# query matrix
print("keys.shape: ",keys.shape)#6*2 since we have 6 words in the sentence and d_out is 2
print("values.shape: ",values.shape)#6*2 since we have 6 words in the sentence and d_out is 2
print("queries.shape: ",queries.shape)#6*2 since we have 6 words in the sentence and d_out is 2
#now lets compute attention scores
#like previously we did for vector embeddings here we will do the same for query and key matrices to get attention scores
#lets calculate for word "journey" which is the second word in the sentence
query_2=queries[1]#query vector for the word journey
attn_scores_2=query_2@keys.T#since dimension of word journey is 1*2 and dimension of keys is 6*2 we need to take transpose of keys to get the attention scores of word journey with all the words in the sentence
print("Attention scores for the word 'journey':",attn_scores_2)
#right now we have not trained the weights hence attn_scores doesnt capture the semantic relationship between the words but after training the weights will be optimized to capture the semantic relationship between the words in the sentence and hence the attention scores will reflect the relevance of each word to the query word "journey" in this case.
#lets do it for whole query matrix
attn_scores=queries@keys.T#matrix multiplication of query matrix with transpose of key matrix to get attention scores for all the words in the sentence
print("Attention scores for all words:\n",attn_scores)
#now lets normalise attention scores to attention weights
#but before that we will scale the attention scores by underrooy(d_key =i.e,2(embedding dimension of keys)) to prevent the attention scores from becoming too large which can lead to numerical instability when we apply softmax function for normalisation
#this is the reason why it is called "Scaled Dot Product Attention"
d_k=keys.shape[-1]
attn_weights_2=torch.softmax(attn_scores_2/d_k**0.5,dim=-1)#normalising attention scores of word journey to get attention weights)
print("Attention weights for the word 'journey':",attn_weights_2)
print(d_k)
#let's print its sum to check if it is 1 since we applied softmax function to normalise attention scores
print(attn_weights_2.sum())
#now lets compute context vector for the word journey by multiplying attention weights with value matrix and summing the results
context_vector_2=0
for i in range(attn_weights_2.shape[0]):
    context_vector_2+=attn_weights_2[i]*values[i]#multiply each value vector with its corresponding attention weight and sum the results to get the context vector for the word journey
print("Context vector for the word 'journey':",context_vector_2)
#or simply we can do it without for loop
context_vector_2_efficient=attn_weights_2@values#matrix multiplication of attention weights with value matrix to get context vector for the word journey more efficiently without using for loop
print("Context vector for the word 'journey' (efficient):",context_vector_2_efficient)
#now lets do it for whole query matrix to get context vectors for all the words in the sentence
attn_weights=torch.softmax(attn_scores/d_k**0.5,dim=-1)
print("Attention weights for all words:\n",attn_weights)#attn_weights is a 6*6 matrix since we have 6 words in the sentence and each word has attention weights with all the words in the sentence including itself
context_vectors=attn_weights@values#matrix multiplication of attention weights with value matrix to get context vectors for all the words in the sentence
print("Context vectors for all words:\n",context_vectors)
#lets accumulate all this in a class in a new python file called Self_attention.py for better understanding and implementation of self attention mechanism with trainable weights(Query,Key,Value) in a more structured way.
from Self_attention import SelfAttention_v1
torch.manual_seed(123)#important to get the same value as before
SA=SelfAttention_v1(3,2)
context_vectors_SA=SA.forward(inputs)
print("Context vectors from SelfAttention class:\n",context_vectors_SA)#it should also print the same as we calculated before
from Self_attention import SelfAttention_v2
torch.manual_seed(789)
SA_v2=SelfAttention_v2(3,2)
context_vectors_SA_v2=SA_v2.forward(inputs)
print("Context vectors from SelfAttention_v2 class:\n",context_vectors_SA_v2)#it should also print the same as we calculated before since we are using the same random seed for weight initialization in both classes
#output differs because v2 uses different initialization scheme for weights and also it uses bias term in the linear transformation of query,key and value matrices which adds some extra values to the context vectors and hence the output differs from v1. 












    
    








