import torch
import torch.nn as nn
import sys
import os

class CNNAE(nn.Module):
    '''
    An implementation of convolutional autoencoder for mnist
    
    Args:
        feat_size (int): Size of extracted image features
    '''
    def __init__(self, feat_size=10):
        super(CNNAE, self).__init__()
        self.feat_size = feat_size
        self.activation = torch.tanh # Set the activation function. Here tanh is recommended.
        
        # Implement each layer to yield the output shape specified in the comments.
        # encoder
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)      # Output Shape [8, 28, 28]
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)     # Output Shape [16, 14, 14]
        self.conv3 = nn.Conv2d(16, 32, 2, padding=0)   # Output Shape [32, 6, 6]
        self.l4 = nn.Linear(32*3*3, 100)    # Output Shape [100]
        self.l5 =  nn.Linear(100, 10)   # Output Shape [feat_size]
        
        # decoder
        self.l6 = nn.Linear(10, 100)    # Output Shape [100]
        self.l7 = nn.Linear(100, 32*3*3)    # Output Shape [288]
        self.conv8 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)    #Output Shape [16, 7, 7]
        self.conv9 =  nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)    #Output Shape [8, 14, 14]
        self.conv10 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2)     #Output Shape [1, 28, 28]
        
        self.pool = nn.MaxPool2d(2,2)

    def encoder(self, x):
        '''
        Extract ``feat_size``-dimensional image features from the input image using nn.Conv2d and nn.Linear.
        The activation function also should be set.
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        # print(f"conv3 shape: {x.shape}")
        
        x = x.view(x.shape[0], -1)
        # print(f"conv3 flatten: {x.shape}")
        x = self.l4(x)
        x = self.activation(x)
        x = self.l5(x)
        hid = self.activation(x)
        # print(f"hid shape: {hid.shape}")
        
        return hid

    def decoder(self, hid):
        '''
        Reconstruct an image from ``feat_size``-dimensional image features using nn.ConvTranspose2d and nn.Linear.
        The activation function also should be set.
        '''
        x = self.l6(hid)
        x = self.activation(x)
        x = self.l7(x)
        x = self.activation(x)
        # print(f"l7 shape: {x.shape}")
        
        x = x.view(x.shape[0],-1, 3, 3)
        # print(f"l7 reshape: {x.shape}")
        x = self.conv8(x)
        x = self.activation(x)
        x = self.conv9(x)
        x = self.activation(x)
        x = self.conv10(x)
        rec_im = self.activation(x)
        
        return rec_im

    def forward(self, im):
        '''
        Declare foward process
        '''
        
        return self.decoder( self.encoder(im) )


if __name__ == "__main__":
    from torchinfo import summary
    
    batch_size = 50
    model = CNNAE()
    summary(model, input_size=(batch_size, 1, 28, 28 ))


