from .clipvip import clipvip32, clipvip16
from torchenhanced import ConfigModule, DevModule
import torch, torch.nn as nn
from .vjepa_reward import MiniBlock

class CLIPVIPReward(ConfigModule):
    """
        Given CLIPVIP weights,
        creates the reward model for videos.

        Expected video shape : (B, 3, T, H, W) (in line with VJEPA, although CLIPVIP expects (B, T, 3, H, W))
    """

    def __init__(self, patch_size=32, clipvip_weights=None, num_frames=12, minihead=False, device='cpu'):
        """
            Args:
                patch_size : int, size of the patches
                clipvip_weights : path to the CLIPVIP weights, if None, random weights are used
                num_frames : number of frames in the video, (pretrained ones are 12)
                minihead : bool, whether to use the minihead or not
                device : str, device to use for the model
        """
        super().__init__()
        
        if(patch_size==32):
            self.clipvip = clipvip32(weights_path=clipvip_weights)
        elif(patch_size==16):
            self.clipvip = clipvip16(weights_path=clipvip_weights)
        else :
            raise ValueError(f'Patch size {patch_size} not supported, use 16 or 32')
    
        self.clipvip.to(device)
        self.clipvip.eval()

        out_tokens, out_dim = self._get_clipvip_output_shape(num_frames)

        if(minihead):
            print(f'Head treats {out_tokens} tokens of dimension {out_dim}')
            # Replace the last two layers (final_conv and classifier)
            minihead_block = MiniBlock(embed_dim=out_dim, num_tokens=out_tokens, device=device)
        else :
            minihead_block = nn.Identity()

        last_linear = nn.Linear(out_dim, 1)

        self.score_head = nn.ModuleDict(dict(minihead=minihead_block, last_linear=last_linear))
        self.minihead = minihead
        self.to(device)

        self.input_shape = (num_frames,3,224,224)

    def body_params(self):
        """
            Returns the parameters of the CLIPVIP model
        """
        return self.clipvip.parameters()

    def head_params(self):
        """
            Returns the parameters of the head of the model
        """
        return self.score_head.parameters()

    def _get_clipvip_output_shape(self, num_frames):
        """
            Returns the shape of the output of the CLIPVIP model
        """
        input_image = torch.randn(1,num_frames,3,224,224, device=self.device)
        output = self.clipvip(input_image)
        last_hidden = output.last_hidden_state[0] # (T,D)

        return last_hidden.shape

    def load_weights(self, path):
        """
            Loads weights of pretrained model
        """
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        """
            Saves the weights of the model
        """
        torch.save(self.state_dict(), path)
        
    def forward(self, x):
        """
            x : (B, T, C, H, W) tensor, correct dimensions for the model

            Returns : (B,) tensor of scores
        """
        B, T, C, H, W = x.shape
        assert C == 3, 'Input must have 3 channels'

        x = self.clipvip(x)

        if(self.minihead):
            x = self.score_head['minihead'](x.last_hidden_state)[:,-1,:] # (B,T,D) keep last token
        else:
            x = x.pooler_output # (B,D) CLIPVIP pooler output

        x = self.score_head['last_linear'](x) # (B,1) Linear head on 


        return x.squeeze(1) # (B,) scores

