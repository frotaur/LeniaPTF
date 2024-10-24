import torch
import torchvision.models as models
from torchenhanced import ConfigModule

class SqueezeReward(ConfigModule):
    """
        Reward model for images based on SqueezeNet.
        Given (B, C, H, W) input, returns (B, 1) of floats.
    """
    def __init__(self, device):
        configo = {}
        super().__init__(configo, device=device)
        
        # Load pretrained SqueezeNet
        squeezenet = models.squeezenet1_1(weights='IMAGENET1K_V1', progress=True)
        
        # SqueezeNet structure, without the head
        self.features = squeezenet.features 

        # Replace the last two layers (final_conv and classifier)
        self.scorer = torch.nn.Linear(512*13*13,1)

        # # Set weights to all ones, for testing :
        # self.scorer.weight.data.fill_(1)
        # self.scorer.bias.data.fill_(0)

    def body_params(self):
        """
            Returns the parameters of the SqueezeNet model
        """
        return self.features.parameters()

    def head_params(self):
        """
            Returns the parameters of the head of the model
        """
        return self.scorer.parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 3, 'Input must have 3 channels'
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        x = self.features(x)

        B, C, H, W = x.shape
        x = self.scorer(x.reshape(B,C*H*W))
        return torch.flatten(x, 1)

if __name__=='__main__':
    model = SqueezeReward('cpu')
    test_img = torch.ones(1, 3, 300, 200)
    print('Test image shape : ', test_img.shape)
    print('Model output shape : ', model(test_img).shape)
    print('model output' , model(test_img).item())
    # print(model)