import torch
import torch.nn as nn

class CNNRNN(nn.Module):
    def __init__(self, rec_size, feat_size=10,trainInitialValue=True):
        super(CNNRNN, self).__init__()

        self.activation = torch.tanh

        # Encoder
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)       #Output Shape [8, 14, 14]
        self.conv2 = nn.Conv2d(8, 16, 4, 2, 1)       #Output Shape [16, 7, 7]
        self.conv3 = nn.Conv2d(16,32, 4, 2, 1)       #Output Shape [32, 2, 2]
        self.l4    = nn.Linear(3*3*32, 100)          #Output Shape [100]
        self.l5    = nn.Linear(100, 10)              #Output Shape [feat_size]

        # RNN
        input_size   = 10+2
        out_vec_size = 2
        out_img_size = 100

        self.h2h = nn.LSTMCell(input_size, rec_size)
        self.h2v = nn.Linear(rec_size, out_vec_size)  # output for vector
        self.h2d = nn.Linear(rec_size, out_img_size)  # output for decoder

        # Decoder
        self.l7     =  nn.Linear(100, 3*3*32)         # Output Shape [288]
        self.conv8  = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1)      #Output Shape [16, 7, 7]
        self.conv9  = nn.ConvTranspose2d(16,  8, 4, 2, padding=1, output_padding=0)                         #Output Shape [8, 14, 14]
        self.conv10 = nn.ConvTranspose2d(8,   3, 4, 2, padding=1, output_padding=0)                         #Output Shape [3, 28, 28]


    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (x.shape[0],-1))
        x = self.l4(x)
        x = self.l5(x)
        return x

    def decoder(self, x):
        x = self.l7(x)
        x = torch.reshape(x, (x.shape[0],-1,3,3))
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        return x

    def forward(self, xi, xv, state=None):
        # Encoder
        im_hid = self.encoder(xi)
        # print(type(im_hid), im_hid.shape)
        
        # RNN
        x = torch.cat((im_hid, xv), dim=1)
        # print(type(x), x.shape)
        new_state = self.h2h(x, state)
        # print(type(new_state), len(new_state), new_state[0].shape)
        out_vec     = self.h2v(new_state[0])
        out_im_feat = self.h2d(new_state[0])
        
        # Decoder
        out_im = self.decoder(out_im_feat)

        return out_im, out_vec, new_state


if __name__ == '__main__':
    from torchinfo import summary
    batch_size = 50

    rnn_model = CNNRNN(rec_size=50)
    summary( rnn_model, input_size=[(batch_size,3,28,28), (batch_size,2)] )
