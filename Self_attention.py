import torch.nn as nn
import torch
class SelfAttention_v1(nn.Module):
    def __init__(self,d_in,d_out):#construct query,key,value matrix(trainable parameters))
        super().__init__()
        self.W_query=nn.Parameter(torch.rand(d_in,d_out))
        self.W_key=nn.Parameter(torch.rand(d_in,d_out))
        self.W_value=nn.Parameter(torch.rand(d_in,d_out))

    def forward(self,x):
        #x is the input sequence of word embeddings with shape (seq_len,embedding_dim)
        queries=x@self.W_query
        keys=x@self.W_key
        values=x@self.W_value
            #lets calculate attn_scores
        attn_scores=queries@keys.T #omega
        #now calculate attn_weights
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        #now compute context vectors by multiplying attn_weights with values

        context=attn_weights@values
        return context
#nn.Linear has an optimized weight initializing scheme and also allows us to easily add bias terms if needed, which can enhance the model's capacity to learn complex patterns in the data. Additionally, using nn.Linear can lead to better performance and faster convergence during training compared to manually defining weight matrices as nn.Parameter.
class SelfAttention_v2(nn.Module):
    def __init__(self,d_in,d_out,qkv_bias=False):
        super().__init__()
        self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)
    
    def forward(self,x):
        keys=self.W_key(x)
        queries=self.W_query(x) 
        values=self.W_value(x)
        attn_scores=queries@keys.T
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)

        context=attn_weights@values 
        return context

    


        
    
        