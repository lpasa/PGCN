from torch.nn import Linear
from torch.nn import functional as F


class Linear_masked_weight(Linear):
    '''
    redefinition of torch.nn.Linear that allows to apply a mask on the weights matrix
    '''
    def forward(self, input, mask):
        maskedW=self.weight*mask
        return F.linear(input, maskedW, self.bias)

