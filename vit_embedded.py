import torch
import torch.nn as nn
import torchsummary
from main import Net
import os

from layers import TransformerEncoder

class ViTEmbedded(nn.Module):
    def __init__(self,args_cnn, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViTEmbedded, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(3136, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )
        """
        1. load the trained cnn model
        self.cnn = ###
        2. set freeze as true
        self.cnn.freeze = true
        """


        args = args_cnn
        args.model_name = 'cnn'
        args.experiment_name = 'cnn_siim'
        net = Net(args)

        net.load_state_dict(torch.load(os.path.join('weights/cnn_siim.pth')), strict=True)
        net.eval()
        # print(*list(net.model.feature_extractor.children())[:-3]) # torch.Size([1024, 64, 56, 56])
        self.cnn =  nn.Sequential(*list(net.model.feature_extractor.children())[:-3])

    def forward(self, x):
        #For Vit
        #torch.Size([18, 3, 32, 32])
        """
        1. call the cnn model
        out = self.cnn(x)
        #we need to make sure that the output is fine for _to_words
        """
        print("ASfdasfa",x.shape)
        out = self.cnn(x)
        print("out",out.shape)
        #torch.Size([18, 64, 8, 8])
        
        out = self._to_words(out)
        print("word",out.shape)
        # for VIT
        #([18, 64, 48])
        # for cnn
        # 18,64,64
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def img_to_patch(x,patch_num):
            """
            Inputs:
                x - torch.Tensor representing the image of shape [B, C, H, W]
                patch_size - Number of pixels per dimension of the patches (integer)

            """
            B, C, H, W = x.shape
            H_new = H // patch_num * patch_num
            W_new = W // patch_num * patch_num
            H_pixels = int(H_new / patch_num)
            W_pixels = int(W_new / patch_num)
            x = x[:, :, :H_new, :W_new]
            #x = x.reshape(B, H_new//H_pixels, H_pixels, W_new//W_pixels, W_pixels, C)
            x = x.reshape(B, C, patch_num, H_pixels, patch_num, W_pixels)
            x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
            x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
            return x