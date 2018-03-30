import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
#required operations
phase = 0
def paranoid_log(x, eps = 1e-8):
    return torch.log(x + eps)

def clamp(x):
    return torch.clamp(x, -8., 8.)

def get_log_alpha(log_sigma2, w):
    log_alpha = clamp(log_sigma2 - paranoid_log(torch.mul(w, w)))
    return log_alpha


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,stride = 1, padding=0, dilation=1, groups = 1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups = groups, bias = bias)
        #get the conv layer parameter shape first
        #construct variable of the same shape with conv layer
        tensor = torch.Tensor(self.weight.shape)
        nn.init.constant(tensor,-10.0)
        self.log_sigma2 = nn.Parameter(torch.Tensor(tensor))
        self.thresh = 3.0

        #
        # #register the log_alphas, then we can collect the paramter and calculate the sparsity
        self.log_alphas = nn.Parameter(torch.Tensor(torch.rand(self.weight.shape)))

    def forward(self, input):
        if phase == 0:
            #warm up
            return self.conv2d_normal(input)
        elif phase == 1:
            #sparse training
            return self.conv2d_noisy(input)
        else:
            #during the testing
            return self.conv2d_masked(input)


    def conv2d_normal(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def conv2d_noisy(self, input):
        ###get the log_alpha first
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        # use the built-in conv2d method to do the convolutional layer calculation
        conved_mu = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        #calculate the noise convoluntional layer
        input_square = torch.mul(input, input)
        w_square = torch.mul(self.weight, self.weight)
        w_square_alpha = torch.exp(log_alpha)*w_square
        conved_si = F.conv2d(input_square, w_square_alpha, None, self.stride, self.padding, self.dilation, self.groups)
        conved_si = torch.sqrt(conved_si + 1e-8)
        #gaussian noise
        Gaussian_noise = Variable(torch.randn(conved_mu.shape)).cuda()
        conved_si = torch.mul(Gaussian_noise, conved_si)

        #record the log_alphas
        self.log_alphas = nn.Parameter(log_alpha.data)

        return conved_mu + conved_si

    def conv2d_masked(self, input):
        # get the select_mask layer
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        select_mask = torch.lt(log_alpha, self.thresh)
        select_mask = select_mask.type(torch.cuda.FloatTensor)

        return F.conv2d(input, torch.mul(self.weight,select_mask), self.bias, self.stride, self.padding, self.dilation)

    def dkl_dp(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

class Full(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Full, self).__init__(in_features, out_features, bias = bias)
        # construct variable of the same shape with full layer
        tensor = torch.Tensor(self.weight.shape)
        nn.init.constant(tensor, -10.0)
        self.log_sigma2 = nn.Parameter(torch.Tensor(tensor))
        self.thresh = 3.0

        # register the log_alphas, then we can collect the paramter and calculate the sparsity
        self.log_alphas = nn.Parameter(torch.Tensor(torch.rand(self.weight.shape)))

    def forward(self, input):
        if phase == 0:
            #warm up
            return self.fc_normal(input)
        elif phase == 1:
            #training time
            return self.fc_noisy(input)
        else:
            #in the testing state
            return self.fc_masked(input)

    def fc_normal(self, input):
        return F.linear(input, self.weight, self.bias)
    def fc_noisy(self, input):
        #get the log_alpha layer
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)

        #use built-in function to calculate the fc laeyr
        mu = F.linear(input, self.weight, self.bias)

        #get the noise layer
        input_square = torch.mul(input, input)
        w_square = torch.mul(self.weight, self.weight)
        w_square_alpha = torch.exp(log_alpha)*w_square
        si = F.linear(input_square, w_square_alpha, None)
        si = torch.sqrt(si + 1e-8)
        #gaussian noise
        Gaussian_noise = Variable(torch.randn(mu.shape)).cuda()
        si = torch.mul(Gaussian_noise, si)

        #record the log_alphas
        self.log_alphas = nn.Parameter(log_alpha.data)
        return mu + si

    def fc_masked(self, input):
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        select_mask = torch.lt(log_alpha, self.thresh)
        select_mask = select_mask.type(torch.cuda.FloatTensor)
        return F.linear(input, torch.mul(self.weight, select_mask), self.bias)

    def dkl_dp(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

#handy check the sparsity
def sparseness(log_alphas, thresh = 3.0):
    N_active = 0.0
    N_total = 0.0
    for la in log_alphas:
        log_alpha_data = log_alphas[la]
        shape = log_alpha_data.shape
        m = torch.lt(log_alpha_data, thresh) # less than, return 1 is la < thresh_tensor
        m = m.type(torch.FloatTensor)# convert byte tensor to float tensor
        n_active = torch.sum(m)
        n_active = n_active.data.numpy()[0]
        n_total = 1.0
        for i in range(len(shape)):
            n_total *= shape[i]

        N_active += n_active
        N_total += n_total
    return 1.0 - N_active / N_total

# #collect the dropout parameters for the sparsity calculation
def gather_logalphas(net):
    log_alphas = {}
    for name, para in net.named_parameters():
        if 'log_alphas' in name:
            log_alphas[name] = para
    return log_alphas
