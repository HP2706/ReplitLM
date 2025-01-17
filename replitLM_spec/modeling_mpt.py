"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
import math
import numpy as np
from .norm import LPLayerNorm
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .attention import attn_bias_shape, build_attn_bias
from .bio import BioLinear
from .blocks import MPTBlock
from .norm import NORM_CLASS_REGISTRY
from .configuration_mpt import MPTConfig
from .adapt_tokenizer import AutoTokenizerForMOD, adapt_tokenizer_for_denoising
from .hf_prefixlm_converter import add_bidirectional_mask_if_missing, convert_hf_causal_lm_to_prefix_lm
from .meta_init_context import init_empty_weights
from .param_init_fns import MODEL_INIT_REGISTRY, generic_param_init_fn_
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = 'model'
    _no_split_modules=["MPTBlock"]

class MPTModel(MPTPreTrainedModel):
    def __init__(self, config: MPTConfig):
        config._validate_config()
        super().__init__(config)
        self.n_layers = config.n_layers
        self.in_dim = config.vocab_size
        self.out_dim = config.vocab_size
        self.n_embed = config.d_model
        
        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).')
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        self.embedding_fraction = config.embedding_fraction
        self.wte = nn.Embedding(config.vocab_size, config.d_model, device=config.init_device)
        if not self.alibi:
            self.wpe = nn.Embedding(config.max_seq_len, config.d_model, device=config.init_device)
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        
        self.blocks = nn.ModuleList([MPTBlock(device=config.init_device, **config.to_dict()) for _ in range(config.n_layers)])
        #for bimt training
     
        #change config back to low_precision_layernorm 
        layernorm_class = LPLayerNorm if config.to_dict()['norm_type'] == "low_precision_layernorm" else nn.LayerNorm
        
        #self.ln_f = layernorm_class(self.n_embed)
        self.l_i = self.get_linear_layers()[1]
        self.l_f = self.get_linear_layers()[-2] #the de_embedding layer
    
        
        # parameters for the bio-inspired trick
        self.l0 = 0.5 # distance between two nearby layers
        self.in_perm = nn.Parameter(torch.tensor(np.arange(int(self.in_dim/self.l_i.in_fold)), dtype=torch.float))
        self.out_perm = nn.Parameter(torch.tensor(np.arange(int(self.out_dim/self.l_f.out_fold)), dtype=torch.float))
        
        self.top_k = 20
        self.res_swap = list(np.arange(2*self.n_layers+1)*3+1)
        self.skip_swap = list(np.arange(2*self.n_layers+1)*3+2)
        self.normal_swap = list(np.arange(2*self.n_layers+2)*3)
   
        self.norm_f = norm_class(config.d_model, device=config.init_device)
        if config.init_device != 'meta':
            print(f'You are using config.init_device={config.init_device!r}, but you can also use config.init_device="meta" with Composer + FSDP for fast initialization.')
            self.apply(self.param_init_fn)
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(self.attn_impl, config.n_heads, config.max_seq_len, self.alibi, prefix_lm=self.prefix_lm, causal=self.is_causal, use_sequence_id=self.attn_uses_sequence_id)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    if config.verbose:
                        warnings.warn(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)
        if config.verbose and config.verbose > 2:
            print(self)
        if 'verbose' not in self.config.init_config:
            self.config.init_config['verbose'] = self.config.verbose
        if self.config.init_config['verbose'] > 1:
            init_fn_name = self.config.init_config['name']
            warnings.warn(f'Using {init_fn_name} initialization.')

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value            
    
    def calculate_sparsity(self):
        non_zero_count = 0
        total_params = 0

        epsilon = 1e-9
        # Loop over the blocks (layers)
        for block in self.blocks:
            for param in block.parameters():
                non_zero_count += torch.sum(torch.abs(param) > epsilon).item()
                total_params += param.numel()

        # Loop over other model components if needed, e.g., embeddings, LayerNorm, etc.
        for param in self.wte.parameters():
            non_zero_count += torch.sum(torch.abs(param) > epsilon).item()
            total_params += param.numel()
        
        if not self.alibi:
            for param in self.wpe.parameters():
                non_zero_count += torch.sum(torch.abs(param) > epsilon).item()
                total_params += param.numel()

        for param in self.emb_drop.parameters():
            non_zero_count += torch.sum(torch.abs(param) > epsilon).item()
            total_params += param.numel()

        for param in self.norm_f.parameters():
            non_zero_count += torch.sum(torch.abs(param) > epsilon).item()
            total_params += param.numel()

        # Calculate the sparsity factor
        sparsity_factor = 1 - (non_zero_count / total_params)

        print(f"Non-zero parameters: {non_zero_count}")
        print(f"Total parameters: {total_params}")
        print(f"Sparsity Factor: {sparsity_factor}")
        return sparsity_factor, non_zero_count, total_params


    @torch.no_grad()
    def _attn_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
                self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias, self.config.n_heads, self.config.max_seq_len, causal=self.is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True
        if self.attn_impl == 'flash':
            return (self.attn_bias, attention_mask)
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        attn_bias = self.attn_bias
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                attn_bias = attn_bias[:, :, :, -s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return (attn_bias, None)

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor):
        (s_k, s_q) = attn_bias.shape[-2:]
        if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
            raise ValueError('attn_bias does not match the expected shape. ' + f'The last two dimensions should both be {self.config.max_length} ' + f'but are {s_k} and {s_q}.')
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor):
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()
        if not return_dict:
            raise NotImplementedError('return_dict False is not implemented yet for MPT')
        if output_attentions:
            raise NotImplementedError('output_attentions is not implemented yet for MPT')
        if attention_mask is not None and attention_mask[:, 0].sum() != attention_mask.shape[0] and self.training:
            raise NotImplementedError('MPT does not support training with left padding.')
        if self.prefix_lm and prefix_mask is None:
            raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError('sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')
        S = input_ids.size(1)
        assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'
        tok_emb = self.wte(input_ids)
        if self.alibi:
            x = tok_emb
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
                past_position = past_key_values[0][0].size(1)
            if S + past_position > self.config.max_seq_len:
                raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
            pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
            assert isinstance(self.emb_drop, nn.Module)
            x = self.emb_drop(x_shrunk)
        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=x.dtype, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]
        all_hidden_states = () if output_hidden_states else None
        for (b_idx, block) in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = past_key_values[b_idx] if past_key_values is not None else None
            (x, past_key_value) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value
        x = self.norm_f(x)
        logits = F.linear(x, self.wte.weight)
        return logits, BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values, hidden_states=all_hidden_states)

    def param_init_fn(self, module):
        init_fn_name = self.config.init_config['name']
        print(f'Initializing {module.__class__.__name__} with {init_fn_name} function {MODEL_INIT_REGISTRY[init_fn_name]} ')
        MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.n_layers, d_model=self.config.d_model, **self.config.init_config)

    def fsdp_wrap_fn(self, module):
        return isinstance(module, MPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, MPTBlock)    
    
    def get_linear_layers(self):
        linear_list = []
        for i in range(self.n_layers -1):
            linear_list = [*linear_list, *self.blocks[i].get_linear_layers()]
        return linear_list    
    
    def get_cc(self, weight_factor=2.0, bias_penalize=True, ln_penalize=True, no_penalize_last=False):
        # compute connection cost
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        cc = 0
        linears = self.get_linear_layers()
        num_linear = len(linears)
        for i in range(num_linear):
            layer = linears[i]
            if isinstance(layer, nn.LayerNorm) or isinstance(layer, LPLayerNorm):
                pass
            else:
                if i == num_linear - 1 and no_penalize_last:
                    weight_factor = 0.
                biolinear = linears[i]
                dist = torch.abs(biolinear.out_coordinates.unsqueeze(dim=1) - biolinear.in_coordinates.unsqueeze(dim=0))
                cc += torch.mean(torch.abs(biolinear.linear.weight).to(device)*(weight_factor*dist+self.l0).to(device))
                if bias_penalize: 
                    if biolinear.linear.bias is not None:
                        cc += torch.mean(torch.abs(biolinear.linear.bias)*(self.l0).to(device))
        return cc
    
    def swap_weight(self, weights, j, k, swap_type="out"):
        with torch.no_grad():  
            if swap_type == "in":
                temp = weights[:,j].clone()
                weights[:,j] = weights[:,k].clone()
                weights[:,k] = temp
            elif swap_type == "out":
                temp = weights[j].clone()
                weights[j] = weights[k].clone()
                weights[k] = temp
            else:
                raise Exception("Swap type {} is not recognized!".format(swap_type))
            
    def swap_bias(self, biases, j, k):
        with torch.no_grad():  
            temp = biases[j].clone()
            biases[j] = biases[k].clone()
            biases[k] = temp
    
    def swap(self, i, j, k):
        # in the ith layer (of neurons), swap the jth and the kth neuron. 
        # Note: n layers of weights means n+1 layers of neurons.
        # (incoming, outgoing) * weights + biases are swapped. 
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i == 0:
            left = None
            right = linears[i]
            self.swap_bias(self.in_perm, j, k)
        elif i == num_linear:
            left = linears[i-1]
            right = None
            self.swap_bias(self.out_perm, j, k)
        else:
            left = linears[i-1]
            right = linears[i]
            
        print(f"left: {type(left)}")    
        print(f"right: {type(right)}")
        
        
        if left != None: 
            if not isinstance(left, nn.LayerNorm) and not isinstance(left, LPLayerNorm):
                print("left of type", type(left))
                fold = left.out_fold
                fold_dim = int(left.linear.weight.shape[0]/fold)
                for l in range(fold):
                    self.swap_weight(left.linear.weight, j+fold_dim*l, k+fold_dim*l, swap_type="out")
                    self.swap_bias(left.linear.bias, j+fold_dim*l, k+fold_dim*l)
                
        if right != None: 
            if not isinstance(right, nn.LayerNorm) and not isinstance(right, LPLayerNorm):
                print("right of type", type(right))
                if i in self.normal_swap:
                    fold = right.in_fold
                    fold_dim = int(right.linear.weight.shape[1]/fold)
                    for l in range(fold):
                        self.swap_weight(right.linear.weight, j+fold_dim*l, k+fold_dim*l, swap_type="in")

                if i in self.res_swap:
                    rightright = linears[i+1]
                    fold = rightright.in_fold
                    fold_dim = int(rightright.linear.weight.shape[1]/fold)
                    for l in range(fold):
                        self.swap_bias(right.bias, j+fold_dim*l, k+fold_dim*l)
                        self.swap_weight(rightright.linear.weight, j+fold_dim*l, k+fold_dim*l, swap_type="in")

            
    def get_score(self, i):
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i == 0:
            left = None
            right = linears[i]
        elif i == num_linear:
            left = linears[i-1]
            right = None
        else:
            left = linears[i-1]
            right = linears[i]
                    
        
        # need to fold attention, fold = 3
               
        
        score = 0.
        if left == None:
            pass
        else:
            if isinstance(left, nn.LayerNorm) or isinstance(left, LPLayerNorm):
                left = linears[i-1]
            fold = left.out_fold
            print(f"fold2: {fold}")
            print(f"right.linear.weight.shape: {left.linear.weight.shape}")
            score +=  torch.mean(torch.sum(torch.abs(left.linear.weight), dim=1).reshape(fold, int(left.linear.weight.shape[0]/fold)), dim=0)
            
        if right == None:
            pass
        else:
            if isinstance(right, nn.LayerNorm) or isinstance(right, LPLayerNorm):
                right = linears[i+1]
            
            fold2 = right.in_fold
            print(f"fold2: {fold2}")
            print(f"right.linear.weight.shape: {right.linear.weight.shape}")
            print(f"score: {score}")
            score += torch.mean(torch.sum(torch.abs(right.linear.weight), dim=0).reshape(fold2, int(right.linear.weight.shape[1]/fold2)), dim=0)
            
        return score
    
    def get_n_head(self, i):
        linears = self.get_linear_layers()
        num_layer = len(linears)
        if i == 0:
            if isinstance(linears[i], nn.LayerNorm) or isinstance(linears[i], LPLayerNorm):
                print("i=0, is norm")
                n_head = linears[i+1].in_head
            else:
                print("i=0, is not norm")
                n_head = linears[i].in_head
        else:
            if isinstance(linears[i-1], nn.LayerNorm) or isinstance(linears[i-1], LPLayerNorm):
                print("i!=0, is norm")
                n_head = linears[i].in_head
            else:
                print("i!=0, is not norm")
                n_head = linears[i-1].out_head
        return n_head

    def get_top_id_head(self, i, top_k=20):
        score = 0.
        if i == self.res_swap[0]:
            for p in self.res_swap:
                score += self.get_score(p)
        else:
            score = self.get_score(i)
            
        n_head = self.get_n_head(i)
        score_head = score.reshape(n_head, int(score.shape[0]/n_head))
        score_head = torch.mean(score_head, dim=1)
        top_id_head = torch.flip(torch.argsort(score_head),[0])[:top_k]
        
        return top_id_head # 1D
    
    
    def get_top_id_tail(self, i, top_k=20):
        score = 0.
        if i == self.res_swap[0]:
            for p in self.res_swap:
                score += self.get_score(p)
        else:
            score = self.get_score(i)
        
        n_head = self.get_n_head(i)
        head_dim = int(score.shape[0]/n_head)
        score_head = score.reshape(n_head, head_dim)
        
        top_id_tail = torch.flip(torch.argsort(score_head, dim=1),[1])[:top_k]
        return top_id_tail # 2D
    
    
    def relocate_ij_head(self, i, j):
        # i-th layer, j-th head
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i < num_linear:
            if isinstance(linears[i], nn.LayerNorm) or isinstance(linears[i], LPLayerNorm):
                num_neuron = int(linears[i+1].linear.weight.shape[1]/linears[i+1].in_fold)
            else:
                num_neuron = int(linears[i].linear.weight.shape[1]/linears[i].in_fold)
        else:
            num_neuron = linears[i-1].linear.weight.shape[0]
            
        ccs = []
                
        num_head = self.get_n_head(i)
        
        head_dim = int(num_neuron/num_head)
        
        for k in range(num_head):
            if i != self.res_swap[0]:
                self.swap(i,head_dim*j+np.arange(head_dim),head_dim*k+np.arange(head_dim))
                
            ccs.append(self.get_cc())
            
            if i != self.res_swap[0]:
                self.swap(i,head_dim*j+np.arange(head_dim),head_dim*k+np.arange(head_dim))
                
        k = torch.argmin(torch.stack(ccs))
        
        if i != self.res_swap[0]:
            self.swap(i,head_dim*j+np.arange(head_dim),head_dim*k+np.arange(head_dim))
        
    
    def relocate_ijk_tail(self, i, j, k):
        # i-th layer, j-th head, k-th neuron
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i < num_linear:
            if isinstance(linears[i], nn.LayerNorm) or isinstance(linears[i], LPLayerNorm):
                num_neuron = int(linears[i+1].linear.weight.shape[1]/linears[i+1].in_fold)
            else:
                num_neuron = int(linears[i].linear.weight.shape[1]/linears[i].in_fold)
        else:
            num_neuron = linears[i-1].linear.weight.shape[0]
            
        ccs = []
        
        def swap_res(j,k):
            for p in self.res_swap:
                self.swap(p,j,k)
                
        num_head = self.get_n_head(i)
        head_dim = int(num_neuron/num_head)
        
        for p in range(head_dim):
            if i == self.res_swap[0]:
                swap_res(j*head_dim+k,j*head_dim+p)
            else:
                self.swap(i,j*head_dim+k,j*head_dim+p)
                
            ccs.append(self.get_cc())
            
            if i == self.res_swap[0]:
                swap_res(j*head_dim+k,j*head_dim+p)
            else:
                self.swap(i,j*head_dim+k,j*head_dim+p)
                
        p = torch.argmin(torch.stack(ccs))
        
        if i == self.res_swap[0]:
                swap_res(j*head_dim+k,j*head_dim+p)
        else:
            self.swap(i,j*head_dim+k,j*head_dim+p)
    
    def relocate_i(self, i):
        # skip swap
        if i in self.skip_swap:
            return
        
        # res swap
        if i in self.res_swap and i != self.res_swap[0]:
            return
            
        # normal swap + the first res swap
        top_id_head = self.get_top_id_head(i, top_k=self.top_k)
        for j in range(top_id_head.item()):
            self.relocate_ij_head(i,j)
            
        top_id_tail = self.get_top_id_tail(i, top_k=self.top_k)
        num_head = top_id_tail.shape[0]
        for j in range(num_head):
            for k in top_id_tail[j]:
                self.relocate_ijk_tail(i,j,k)
            
    def relocate(self):
        print('swap')
        # Relocate neurons in the whole model
        linears = self.get_linear_layers()
        num_linear = len(linears)
        for i in range(num_linear-1):
            #print(i)
            self.relocate_i(i)
    
    

class MPTForCausalLM(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')
        self.transformer = MPTModel(config)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale
            
            
    def get_linear_layers(self):
        self.transformer.get_linear_layers()
    
    def swap_weight(self, weights, j, k, swap_type="out"):
        self.transformer.swap_weight(weights, j, k, swap_type="out")
            
    def swap_bias(self, biases, j, k):
        self.transformer.swap_bias(biases, j, k)
    
    def swap(self, i, j, k):
        self.transformer.swap(i, j, k)
        
    def get_score(self, i):
        self.transformer.get_score(i)
        
    def get_n_head(self, i):
        self.transformer.get_n_head(i)

    def get_top_id_head(self, i, top_k=20):
        self.transformer.get_top_id_head(i, top_k = 20)       

    def get_top_id_tail(self, i):
        self.transformer.get_top_id_tail(i, top_k = 20)
    
    def relocate_ij_head(self, i, j):
        self.transformer.relocate_ij_head(i, j)
    
    def relocate_ijk_tail(self, i, j, k):
        self.transformer.relocate_ijk_tail(i, j, k)
    
    def relocate_i(self, i):
        self.transformer.relocate_i(i)
        
    def relocate(self):
        self.transformer.relocate()
        
    def get_cc(self):
        return self.transformer.get_cc()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.transformer.wte

    def set_output_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache)
        logits = F.linear(outputs.last_hidden_state, self.transformer.wte.weight)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def param_init_fn(self, module):
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.n_layers, d_model=self.config.d_model, **self.config.init_config)

    def fsdp_wrap_fn(self, module):
        return isinstance(module, MPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, MPTBlock)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds is not implemented for MPT yet')
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get('use_cache') == False:
                raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'prefix_mask': prefix_mask, 'sequence_id': sequence_id, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache', True)}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [tuple((past_state.index_select(0, beam_idx) for past_state in layer_past))]
        return reordered_past
