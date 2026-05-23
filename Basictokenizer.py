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
torch.manual_seed(123)
embedding_layer=torch.nn.Embedding(vocab_size,output_dim)#Embedding creates a dictionary(look-up table) and initialise all the weights of the embeddings matrix randomly 
print(embedding_layer.weight)
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
token_embedding_layer=torch.nn.Embedding(vocab_size,output_dim)
max_length=4
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False)
data_iter=iter(dataloader)#dataloader helps us to manage the task of inputing,batching,creating different batches,parallel processing becomes more easier
inputs,targets=next(data_iter)
print("Token ID's:\n",inputs)
print("\nInputs shape:\n",inputs.shape)
#8*4 dimensional token id tensor since batch_size=8 and max_length i.e,context_size=4
#for every input id in 8*4 tensor lets get 1 row(256 dimensions/cols) that is vector embedding layer(look-up table)
token_embeddings=token_embedding_layer(inputs)
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
inputs=torch.tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]])
query=inputs[1]
attn_scores_2=torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i]=torch.dot(x_i,query)
print(attn_scores_2)#printing attention scores of query w.r.t every input embedding





    
    








