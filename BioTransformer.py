
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import math


seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

torch.set_default_tensor_type(torch.DoubleTensor)


class BioLinear(nn.Module):

    def __init__(self, in_dim, out_dim, in_fold=1, out_fold=1, in_head=1, out_head=1):
        super(BioLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_fold = in_fold
        self.out_fold = out_fold
        self.in_head = in_head
        self.out_head = out_head
        assert in_dim % in_fold == 0
        assert out_dim % out_fold == 0
        #compute in_cor, shape: (in_dim)
        in_dim_fold = int(in_dim/in_fold)
        out_dim_fold = int(out_dim/out_fold)
        self.in_coordinates = torch.tensor(list(np.linspace(1/(2*in_dim_fold), 1-1/(2*in_dim_fold), num=in_dim_fold))*in_fold, dtype=torch.float)
        self.out_coordinates = torch.tensor(list(np.linspace(1/(2*out_dim_fold), 1-1/(2*out_dim_fold), num=out_dim_fold))*out_fold, dtype=torch.float)
        
    def forward(self, x):
        return self.linear(x)
class BioMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, w=2, depth=2, shp=None):
        super(BioMLP, self).__init__()
        if shp == None:
            shp = [in_dim] + [w]*(depth-1) + [out_dim]
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.depth = depth
                 
        else:
            self.in_dim = shp[0]
            self.out_dim = shp[-1]
            self.depth = len(shp) - 1
        linear_list = []
        for i in range(self.depth):
            linear_list.append(BioLinear(shp[i], shp[i+1]))
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
    

    def forward(self, x):
        f = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        #f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = f(self.linears[i](x))
        x = self.linears[-1](x)
        return x
    
    def get_linear_layers(self):
        return self.linears
class BioAttention(nn.Module):

    def __init__(self, n_head=2, n_embed=6):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.l_attn = BioLinear(n_embed, 3*n_embed, out_fold=3, out_head=n_head)
        # output projection
        self.l_proj = BioLinear(n_embed, n_embed, in_fold=1, in_head=n_head)
        #self.l_proj = BioLinear(n_embed, n_embed)
        # regularization
        self.n_head = n_head
        self.n_embed = n_embed

    def forward(self, x):
        # B: batch size; T: sequence length; C: embedding dimensionality (n_embd)
        B, T, C = x.size()

        # query, key, value
        x = self.l_attn(x)
        q, k, v = x[:,:,:C], x[:,:,C:2*C], x[:,:,2*C:3*C]
        n_head = self.n_head
        assert C % n_head == 0
        q = q.reshape(B, T, n_head, int(C/n_head))
        k = k.reshape(B, T, n_head, int(C/n_head))
        v = v.reshape(B, T, n_head, int(C/n_head))

        # (causal) self-attention
        attn = torch.einsum('ijhl,ikhl->ijkh', q, k)/np.sqrt(int(C/n_head))
        mask = torch.ones(T,T)*float('-inf')
        mask = torch.tril(mask, diagonal=-1).permute(1,0).unsqueeze(dim=0).unsqueeze(dim=3)
        attn = attn + mask
        attn = nn.Softmax(dim=2)(attn)
        attn = torch.einsum('ijkl,iklh->ijlh', attn, v)
        attn = attn.reshape(B, T, C)

        # output projection
        y = self.l_proj(attn)
        return y
    
    def get_linear_layers(self):
        return [self.l_attn, self.l_proj]
class BioBlock(nn.Module):
    # A transformer block
    def __init__(self, n_head=2, n_embed=6):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = BioAttention(n_head=n_head, n_embed=n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = BioMLP(shp=[n_embed, 4*n_embed, n_embed])

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    
    def get_linear_layers(self):
        return [self.ln_1, *self.attn.get_linear_layers(), self.ln_2, *self.mlp.get_linear_layers()]
class BioTransformer(nn.Module):
    # Transformer: since our goal is to deal with linear regression, not language, 
    # we ignore token embeddings and positioanl embeddings. 
    def __init__(self, in_dim=3, out_dim=3, n_head=2, n_embed=20, n_layer=2, block_size=19):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.l_i = BioLinear(in_dim, n_embed)
        self.blocks = nn.ModuleList([BioBlock(n_head=n_head, n_embed=n_embed) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        #self.ln_f.weight.requires_grad = False
        #self.ln_f.bias.requires_grad = False
        self.l_f = BioLinear(n_embed, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        #self.pe = torch.nn.Parameter(torch.normal(0,1,size=(block_size, n_embed)))
        
        # parameters for the bio-inspired trick
        self.l0 = 0.5 # distance between two nearby layers
        #self.in_perm = torch.tensor(np.arange(int(self.in_dim/self.l_i.in_fold)), dtype=torch.long)
        self.in_perm = nn.Parameter(torch.tensor(np.arange(int(self.in_dim/self.l_i.in_fold)), dtype=torch.float))
        #self.out_perm = torch.tensor(np.arange(int(self.out_dim/self.l_f.out_fold)), dtype=torch.long)
        self.out_perm = nn.Parameter(torch.tensor(np.arange(int(self.out_dim/self.l_f.out_fold)), dtype=torch.float))
        self.top_k = 20
        
        
        self.res_swap = list(np.arange(2*n_layer+1)*3+1)
        self.skip_swap = list(np.arange(2*n_layer+1)*3+2)
        self.normal_swap = list(np.arange(2*n_layer+2)*3)
        

    def forward(self, x):
        x = x[:,:,self.in_perm.long()]
        x = self.l_i(x)
        #x = x + self.pe.unsqueeze(dim=0) # positional encoding
        for i in range(self.n_layer):
            x = self.blocks[i](x)
        #x = self.ln_f(x)
        y = self.l_f(x)
        
        out_perm_inv = torch.zeros(self.out_dim, dtype=torch.long)
        out_perm_inv[self.out_perm.long()] = torch.arange(self.out_dim)
        y = y[:,:,out_perm_inv]
        return y
    
    def get_linear_layers(self):
        linear_list = [self.l_i]
        for i in range(self.n_layer):
            linear_list = [*linear_list, *self.blocks[i].get_linear_layers()]
        linear_list.append(self.ln_f)
        linear_list.append(self.l_f)
        return linear_list
    
    
    def get_cc(self, weight_factor=2.0, bias_penalize=True, ln_penalize=True, no_penalize_last=False):
        # compute connection cost
        cc = 0
        linears = self.get_linear_layers()
        num_linear = len(linears)
        for i in range(num_linear):
            layer = linears[i]
            if isinstance(layer, nn.LayerNorm):
                pass
                #cc += torch.sum(torch.abs(layer.weight)) + torch.sum(torch.abs(layer.bias))
            else:
                if i == num_linear - 1 and no_penalize_last:
                    weight_factor = 0.
                biolinear = linears[i]
                dist = torch.abs(biolinear.out_coordinates.unsqueeze(dim=1) - biolinear.in_coordinates.unsqueeze(dim=0))
                cc += torch.mean(torch.abs(biolinear.linear.weight)*(weight_factor*dist+self.l0))
                if bias_penalize == True:
                    cc += torch.mean(torch.abs(biolinear.linear.bias)*(self.l0))
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
            
        
        if left != None:
            fold = left.out_fold
            fold_dim = int(left.linear.weight.shape[0]/fold)
            for l in range(fold):
                self.swap_weight(left.linear.weight, j+fold_dim*l, k+fold_dim*l, swap_type="out")
                self.swap_bias(left.linear.bias, j+fold_dim*l, k+fold_dim*l)
                
        if right != None:
        
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
                    self.swap_bias(right.weight, j+fold_dim*l, k+fold_dim*l)
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
        
        if isinstance(right, nn.LayerNorm):
            right = linears[i+1]
            
        # need to fold attention, fold = 3
        score = 0.
        if left == None:
            pass
        else:
            fold = left.out_fold
            score +=  torch.mean(torch.sum(torch.abs(left.linear.weight), dim=1).reshape(fold, int(left.linear.weight.shape[0]/fold)), dim=0)
            
        if right == None:
            pass
        else:
            fold2 = right.in_fold
            score += torch.mean(torch.sum(torch.abs(right.linear.weight), dim=0).reshape(fold2, int(right.linear.weight.shape[1]/fold2)), dim=0)
            
        return score
    
    def get_n_head(self, i):
        linears = self.get_linear_layers()
        num_layer = len(linears)
        if i == 0:
            n_head = linears[0].in_head
        else:
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
            if isinstance(linears[i], nn.LayerNorm):
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
            if isinstance(linears[i], nn.LayerNorm):
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
        for j in top_id_head:
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
        for i in range(num_linear+1):
            #print(i)
            self.relocate_i(num_linear-i)
            
            
    def plot(self):
        layers = self.get_linear_layers()
        linears = [layers[i] for i in [0,2,3,5,6,8,9,11,12,14]]
        shp = [2,32,32,32,128,32,32,32,128,32,2]

        fig, ax = plt.subplots(figsize=(5,10))
        s = 1/(2*max(shp))
        for j in range(len(shp)):
            N = shp[j]
            for i in range(N):
                circle = Ellipse((1/(2*N)+i/N, 0.1*j), s, s/15*((len(shp)-1)+0.4), color='black')
                ax.add_patch(circle)


        plt.ylim(-0.02,0.1*(len(shp)-1)+0.02)
        plt.xlim(-0.02,1.02)

        #linears = self.linears
        ln_id = [1,4,7,10,13]
        jj = 0
        for ii in range(len(linears)):
            biolinear = linears[ii]
            p = biolinear.linear.weight.clone()
            if ii in [1,3,5,7,9]:
                ln = layers[ln_id[jj]]
                p = p * ln.weight.clone().unsqueeze(dim=0)
                jj += 1
            p_shp = p.shape
            if ii == 1 or ii == 5:
                p_ = p[64:,:]
                #p_ = p[8:,:]
                p = p/torch.abs(p_).max()
            else:
                p = p/torch.abs(p[:,:]).max()
            fold_num = int(p_shp[1])
            if ii == 1 or ii == 5:
                p_shp0 = int(p_shp[0]/3)
                for i in range(p_shp0):
                    for j in range(fold_num):
                        plt.plot([1/(2*p_shp0)+i/p_shp0, 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii], lw=np.minimum(10*np.abs(p[i+64,j].detach().numpy()), 1), color="red" if p[i,j]>0 else "blue")
                        #plt.plot([1/(2*p_shp0)+i/p_shp0, 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii], lw=np.minimum(10*np.abs(p[i+8,j].detach().numpy()), 1), color="red" if p[i,j]>0 else "blue")

            else:
                for i in range(p_shp[0]):
                    for j in range(fold_num):
                        plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii], lw=np.minimum(10*np.abs(p[i,j].detach().numpy()), 1), color="red" if p[i,j]>0 else "blue")


        ax.axis('off')

        fontsize=12
        plt.text(1.1,0.0, "input", fontsize=fontsize)
        plt.text(1.1,0.1, "LN", fontsize=fontsize)
        plt.text(1.1,0.2, "Atn", fontsize=fontsize)
        plt.text(1.1,0.3, "Res", fontsize=fontsize)
        plt.text(1.1,0.4, "MLP hidden", fontsize=fontsize)
        plt.text(1.1,0.5, "Res", fontsize=fontsize)
        plt.text(1.1,0.6, "Atn", fontsize=fontsize)
        plt.text(1.1,0.7, "Res", fontsize=fontsize)
        plt.text(1.1,0.8, "MLP hidden", fontsize=fontsize)
        plt.text(1.1,0.9, "Res", fontsize=fontsize)
        plt.text(1.1,1.0, "output", fontsize=fontsize)
import numpy as np


seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
# create linear regression dataset: (x_1, x2, ... ,x_d) -> y
# each sequence has length T (T sample)
d = 1
T = 2

n_w = 10000
#n_w = 10
w = torch.rand(n_w, d)*2 - 1
#torch.normal(0,1,size=(n_w, d))

# x has shape (n_w, T, d)
#x = torch.normal(0,1,size=(n_w, T, d))
x = torch.rand(n_w, T, d)*2 + 1
y = torch.einsum('ik,ijk->ij', w, x)
y_ = torch.cat([y, torch.zeros([y.shape[0], y.shape[1]*d])], dim=1).reshape(n_w, d+1, T).permute(0,2,1)
x_ = torch.cat([torch.zeros([y.shape[0], y.shape[1], 1]), x], dim=2)
data = torch.cat([x_, y_], dim=2)
data = data.reshape(n_w, -1)
inputs = data[:,:(2*T-1)*(d+1)].reshape(n_w, 2*T-1, d+1)
labels = data[:,(d+1):].reshape(n_w, 2*T-1, d+1)

fraction = 0.8
train_num = int(n_w*fraction)
test_num = n_w - train_num

train_id = np.random.choice(n_w,train_num,replace=False)
test_id = np.array(list(set(np.arange(n_w)) - set(train_id)))

inputs_train = inputs[train_id].requires_grad_(True)
labels_train = labels[train_id]

inputs_test = inputs[test_id].requires_grad_(True)
labels_test = labels[test_id]

### train ###
model = BioTransformer(in_dim=d+1, out_dim=d+1, n_head=1, n_embed=32, n_layer=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
steps = int(40000)
log = 100
batch_size = 64 #128
lamb = 1e-3
swap_log = int(1e6) #1000
plot_log = 1000


for step in range(steps):
    
    if step == int(steps*1/4):
        lamb *= 10
    
    if step == int(steps*3/4):
        lamb *= 10
    
    
    optimizer.zero_grad()
    
    if batch_size == None:
        train_choice = np.arange(train_num)
        test_choice = np.arange(test_num)
    else:
        train_choice = np.random.choice(train_num, batch_size, replace=True)
        test_choice = np.random.choice(test_num, batch_size, replace=True)
        
    pred  = model(inputs_train[train_choice])
    diff = pred[:,::2,0] - labels_train[train_choice][:,::2,0]
    loss = torch.mean(diff**2)
    loss_last = torch.mean(diff[:,-1]**2)
    
    pred_test  = model(inputs_test[test_choice])
    diff_test = pred_test[:,::2,0] - labels_test[test_choice][:,::2,0]
    loss_test = torch.mean(diff_test**2)
    loss_test_last = torch.mean(diff_test[:,-1]**2)
    
    cc = model.get_cc(no_penalize_last=False)
    total_loss = loss + lamb*cc
    total_loss.backward()

    
    optimizer.step()
    
    if step % log == 0:
        print("step = %d | train loss: %.2e | train last: %.2e | test loss %.2e | test last: %.2e | cc: %.2e "%(step, loss.detach().numpy(), loss_last.detach().numpy(), loss_test.detach().numpy(), loss_test_last.detach().numpy(), cc.detach().numpy()))
        
    if (step+1) % swap_log == 0:
        model.relocate()
        
    if (step+1) % plot_log == 0:
        model.plot()
        plt.show()
        plt.scatter(pred[:,-1,0].detach().numpy(), labels_train[train_choice][:,-1,0].detach().numpy(), s=5)
        plt.show()

    
layers = model.get_linear_layers()
linears = [layers[i] for i in [0,2,3,5,6,8,9,11,12,14]]
shp = [2,32,32,32,128,32,32,32,128,32,2]
#shp = [2,4,4,4,16,4,4,4,16,4,2]
#shp = [2,16,16,16,64,16,16,16,64,16,2]

fig, ax = plt.subplots(figsize=(5,10))
s = 1/(2*max(shp))
for j in range(len(shp)):
    N = shp[j]
    for i in range(N):
        circle = Ellipse((1/(2*N)+i/N, 0.1*j), s, s/15*((len(shp)-1)+0.4), color='black')
        ax.add_patch(circle)


plt.ylim(-0.02,0.1*(len(shp)-1)+0.02)
plt.xlim(-0.02,1.02)

#linears = self.linears
ln_id = [1,4,7,10,13]
jj = 0
for ii in range(len(linears)):
    biolinear = linears[ii]
    p = biolinear.linear.weight.clone()
    if ii in [1,3,5,7,9]:
        ln = layers[ln_id[jj]]
        p = p * ln.weight.clone().unsqueeze(dim=0)
        jj += 1
    p_shp = p.shape
    if ii == 1 or ii == 5:
        p_ = p[64:,:]
        p = p/torch.abs(p_).max()
    else:
        p = p/torch.abs(p[:,:]).max()
    fold_num = int(p_shp[1])
    if ii == 1 or ii == 5:
        p_shp0 = int(p_shp[0]/3)
        for i in range(p_shp0):
            for j in range(fold_num):
                plt.plot([1/(2*p_shp0)+i/p_shp0, 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii], lw=np.minimum(10*np.abs(p[i+64,j].detach().numpy()), 1), color="blue" if p[i,j]>0 else "red")

    else:
        for i in range(p_shp[0]):
            for j in range(fold_num):
                plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii], lw=np.minimum(10*np.abs(p[i,j].detach().numpy()), 1), color="blue" if p[i,j]>0 else "red")

ax.axis('off')

fontsize=12
shift = 0.03
plt.text(1.1,0.0-shift, "input", fontsize=fontsize, rotation=90)
plt.text(1.1,0.1-shift, "Embed", fontsize=fontsize, rotation=90)
plt.text(1.1,0.2-shift, "Atn1", fontsize=fontsize, rotation=90)
plt.text(1.1,0.3-shift, "Res1", fontsize=fontsize, rotation=90)
plt.text(1.1,0.4-shift, "   MLP1 \n hidden", fontsize=fontsize, rotation=90)
plt.text(1.1,0.5-shift, "Res2", fontsize=fontsize, rotation=90)
plt.text(1.1,0.6-shift, "Atn2", fontsize=fontsize, rotation=90)
plt.text(1.1,0.7-shift, "Res3", fontsize=fontsize, rotation=90)
plt.text(1.1,0.8-shift, "   MLP2 \n hidden", fontsize=fontsize, rotation=90)
plt.text(1.1,0.9-shift, "Res4", fontsize=fontsize, rotation=90)
plt.text(1.1,1.0-shift, "output", fontsize=fontsize, rotation=90)

#circle = Ellipse((0.47,0.5), 20*s, 10*s/15*((len(shp)-1)+0.4), color='black', fill=False, linewidth=4)
#ax.add_patch(circle)
for i in range(32):
    plt.text(i/32+0.007, 0.51, i+1, fontsize=6, rotation=90)
    plt.text(i/32+0.007, 0.31, i+1, fontsize=6, rotation=90)
    plt.text(i/32+0.007, 0.71, i+1, fontsize=6, rotation=90)
    plt.text(i/32+0.007, 0.91, i+1, fontsize=6, rotation=90)
    
plt.text(0.23,-0.02,0, fontsize=fontsize, rotation=90)
plt.text(0.73,-0.02,"x", fontsize=fontsize, rotation=90)
plt.text(0.23,1.02,"y", fontsize=fontsize,rotation=90)
plt.text(0.73,1.02,0, fontsize=fontsize,rotation=90)

#plt.savefig('./results/incontext2/incontext.png', bbox_inches="tight")
x = inputs_train
x = x[:,:,model.in_perm.long()]
x = model.l_i(x)


B, T, C = x.size()
# atn1
xx = model.blocks[0].attn.l_attn(x)
q, k, v = xx[:,:,:C], xx[:,:,C:2*C], xx[:,:,2*C:3*C]
n_head = model.blocks[0].attn.n_head
assert C % n_head == 0
q = q.reshape(B, T, n_head, int(C/n_head))
k = k.reshape(B, T, n_head, int(C/n_head))
v = v.reshape(B, T, n_head, int(C/n_head))

# (causal) self-attention
attn = torch.einsum('ijhl,ikhl->ijkh', q, k)/np.sqrt(int(C/n_head))
mask = torch.ones(T,T)*float('-inf')
mask = torch.tril(mask, diagonal=-1).permute(1,0).unsqueeze(dim=0).unsqueeze(dim=3)
attn = attn + mask
attn = nn.Softmax(dim=2)(attn)
attn = torch.einsum('ijkl,iklh->ijlh', attn, v)
attn = attn.reshape(B, T, C)

# output projection
y = model.blocks[0].attn.l_proj(attn)

print(x.shape, y.shape)
x = x + y

#prob = x.clone()

f = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
y = f(model.blocks[0].mlp.linears[0](x))
y = model.blocks[0].mlp.linears[1](y)

x = x + y

prob = x.clone()

B, T, C = x.size()
# atn1
xx = model.blocks[1].attn.l_attn(x)
q, k, v = xx[:,:,:C], xx[:,:,C:2*C], xx[:,:,2*C:3*C]
n_head = model.blocks[1].attn.n_head
assert C % n_head == 0
q = q.reshape(B, T, n_head, int(C/n_head))
k = k.reshape(B, T, n_head, int(C/n_head))
v = v.reshape(B, T, n_head, int(C/n_head))

# (causal) self-attention
attn = torch.einsum('ijhl,ikhl->ijkh', q, k)/np.sqrt(int(C/n_head))
mask = torch.ones(T,T)*float('-inf')
mask = torch.tril(mask, diagonal=-1).permute(1,0).unsqueeze(dim=0).unsqueeze(dim=3)
attn = attn + mask
attn = nn.Softmax(dim=2)(attn)
attn = torch.einsum('ijkl,iklh->ijlh', attn, v)
attn = attn.reshape(B, T, C)

#prob = attn.clone()


# output projection
y = model.blocks[1].attn.l_proj(attn)

x = x + y


#prob = x.clone()

f = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
y = f(model.blocks[1].mlp.linears[0](x))
y = model.blocks[1].mlp.linears[1](y)

x = x + y




y = model.l_f(x)

out_perm_inv = torch.zeros(model.out_dim, dtype=torch.long)
out_perm_inv[model.out_perm.long()] = torch.arange(model.out_dim)
y = y[:,:,out_perm_inv]
model.get_top_id_tail(10)
for iid in range(32):
    plt.scatter(inputs_train[:,-1,1].detach().numpy(), prob[:,-1,iid].detach().numpy(), s=0.1)
    plt.scatter(w[train_id,0].detach().numpy(), prob[:,-1,iid].detach().numpy(), s=0.1)
    plt.scatter(w[train_id,0].detach().numpy()*inputs_train[:,-1,1].detach().numpy(), prob[:,-1,iid].detach().numpy(), s=0.1)
    print(iid)
    plt.show()
id1 = 8
id2 = 11
plt.scatter(prob[:,-1,id1].detach().numpy(), prob[:,-1,id2].detach().numpy(), s=0.1, c= w[train_id,0].detach().numpy())
plt.figure(figsize=(6,18))

plt.subplot(3,1,1)

pred  = model(inputs)

plt.scatter(labels[:,-1,0].detach().numpy(),pred[:,-1,0].detach().numpy(), s=0.1)
plt.plot([-3,3],[-3,3], ls="--",color="red",alpha=0.7)
plt.xlabel('True '+r"$y$",fontsize=25)
plt.ylabel('Predicted '+r"$y$",fontsize=25)

plt.subplot(3,1,2)
id1 = 8
id2 = 9
plt.scatter(prob[:,-1,id1].detach().numpy(), prob[:,-1,id2].detach().numpy(), s=0.1, c= w[train_id,0].detach().numpy())
cbar = plt.colorbar()
plt.xlabel("Neuron 9", fontsize=25)
plt.ylabel("Neuron 10", fontsize=25)
cbar.ax.set_ylabel("weight scalar",fontsize=25)

plt.subplot(3,1,3)
id1 = 10
id2 = 18
plt.scatter(prob[:,-1,id1].detach().numpy(), prob[:,-1,id2].detach().numpy(), s=0.1, c= w[train_id,0].detach().numpy())
cbar = plt.colorbar()
plt.xlabel("Neuron 11", fontsize=25)
plt.ylabel("Neuron 19", fontsize=25)
cbar.ax.set_ylabel("weight scalar",fontsize=25)

plt.savefig("./results/incontext2/weight_neuron.png", bbox_inches="tight")
 