import numpy as np
import math
from code_data.layer_utils import *


class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=7, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32, use_batchnorm=False, num_of_conv_layers=1, stride_conv=1, momentum=0.9, eps=1e-5, use_pool=None, pool_size_stride={'pool_height': 2, 'pool_width': 2, 'stride': 2}, hidden_layer_dimensions=[4096]):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        self.num_of_conv_layers = num_of_conv_layers
        self.stride_conv = stride_conv
        self.pool_param = pool_size_stride
        self.hidden_layer_dimensions = hidden_layer_dimensions

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        if use_pool is None:
            use_pool = [True]*num_of_conv_layers
        # Size of the input
        self.pool_param.update({'use_pool':use_pool})
        C, H, W = input_dim

        Hp=0
        Wp=0
        F=0
        for layer_count in range(num_of_conv_layers):
            F = num_filters[layer_count]
            pad = (filter_size - 1) / 2
            Hc = (H + 2 * pad - filter_size) / stride_conv + 1
            Wc = (W + 2 * pad - filter_size) / stride_conv + 1

            self.params.update({'W_'+str(layer_count+1): weight_scale * np.random.randn(F, C, filter_size, filter_size),'b_'+str(layer_count+1): np.zeros(F)})
            
            if self.use_batchnorm:
                bn_param = {'mode': 'train','running_mean': np.zeros(F),'running_var': np.zeros(F), 'eps': eps, 'momentum': momentum}
                gamma = np.ones(F)
                beta = np.zeros(F)
                self.bn_params.update({'bn_param_'+str(layer_count+1): bn_param})
                self.params.update({'beta_'+str(layer_count+1): beta, 'gamma_'+str(layer_count+1): gamma})

            if use_pool[layer_count]:
                Hp = (Hc - int(pool_size_stride['pool_height'])) / int(pool_size_stride['stride']) + 1
                Wp = (Wc - int(pool_size_stride['pool_width'])) / int(pool_size_stride['stride']) + 1
            else:
                Hp = Hc
                Wp = Wc

            H = Hp
            W = Wp
            C=F
        
        previous_dim = F * Hp * Wp
        for hidden_dim_idx,hidden_dim in enumerate(hidden_layer_dimensions):
            Hh = hidden_dim
            W2 = weight_scale * np.random.randn(previous_dim, Hh)
            b2 = np.zeros(Hh)
            self.params.update({'W2_'+str(hidden_dim_idx): W2,'b2_'+str(hidden_dim_idx): b2})
            previous_dim = Hh
            if self.use_batchnorm:
                bn_param = {'mode': 'train','running_mean': np.zeros(Hh),'running_var': np.zeros(Hh)}
                gamma = np.ones(Hh)
                beta = np.zeros(Hh)
                
                self.bn_params.update({'bn_param2_'+str(hidden_dim_idx): bn_param})
                
                self.params.update({'beta2_'+str(hidden_dim_idx): beta,'gamma2_'+str(hidden_dim_idx): gamma})

        Hc = num_classes
        W3 = weight_scale * np.random.randn(Hh, Hc)
        b3 = np.zeros(Hc)

        self.params.update({'W3': W3,'b3': b3})


        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        N = X.shape[0]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W_1'].shape[2]
        conv_param = {'stride': self.stride_conv, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        #pool_param = {'pool_height': set_pool_height, 'pool_width': set_pool_width, 'stride': set_pool_stride, 'use_pool': self.params['use_pool']}
        #pool_param = self.pool_details['pool_param']

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################
        conv_layer = X
        cache_conv_layer = []
        for conv_layer_num in range(self.num_of_conv_layers):
            w, b = self.params['W_'+str(conv_layer_num+1)], self.params['b_'+str(conv_layer_num+1)]
            # Forward into the conv layer
            if self.use_batchnorm:
                bn_param = self.bn_params['bn_param_'+str(conv_layer_num+1)]
                gamma = self.params['gamma_'+str(conv_layer_num+1)]
                beta = self.params['beta_'+str(conv_layer_num+1)]
                print conv_layer.shape
                conv_layer, cache_conv_layer_ = conv_norm_relu_pool_forward(conv_layer, w, b, conv_param, self.pool_param, gamma, beta, bn_param, conv_layer_num)
                print conv_layer.shape
            else:
                conv_layer, cache_conv_layer_ = conv_relu_pool_forward(conv_layer, w, b, conv_param, self.pool_param, conv_layer_num)
            cache_conv_layer.append(cache_conv_layer_)
            print 'conv_layer_num', conv_layer_num

        N, F, Hp, Wp = conv_layer.shape  # output shape

        print 'forward into the hidden layer'
        #W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        x = conv_layer.reshape((N, F * Hp * Wp))

        hidden_layer_cache = []
        hidden_layer_dimensions = self.hidden_layer_dimensions
        for hidden_dim_idx,hidden_dim in enumerate(hidden_layer_dimensions):
            w,b = self.params['W2_'+str(hidden_dim_idx)], self.params['b2_'+str(hidden_dim_idx)]
            if self.use_batchnorm:
                bn_param = self.bn_params['bn_param2_'+str(hidden_dim_idx)]
                gamma = self.params['gamma2_'+str(hidden_dim_idx)]
                beta = self.params['beta2_'+str(hidden_dim_idx)]
                hidden_layer, hidden_layer_cache_ = affine_norm_relu_forward(x, w, b, gamma, beta, bn_param)
            else:
                hidden_layer, hidden_layer_cache_ = affine_relu_forward(x, w, b)
            hidden_layer_cache.append(hidden_layer_cache_)
            N, Hh = hidden_layer.shape
            x = hidden_layer
            print 'hidden layer', hidden_dim_idx
        
        print 'forward into the output layer'
        
        w = W3
        b = b3
        scores, cache_scores = affine_forward(x, w, b)

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        print 'loss function'
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for conv_layer_num in range(self.num_of_conv_layers):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W_'+str(conv_layer_num+1)]**2)
        for hidden_dim_idx in range(len(hidden_layer_dimensions)):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W2_'+str(hidden_dim_idx)]**2)
        reg_loss += 0.5 * self.reg * np.sum(W3**2)
        loss = data_loss + reg_loss


        print 'backprop start'
        grads = {}
        dx3, dW3, db3 = affine_backward(dscores, cache_scores)
        dW3 += self.reg * W3
        grads.update({'W3': dW3, 'b3': db3})

        dx2 = dx3
        dW2 = None
        for hidden_dim_idx in range(len(hidden_layer_dimensions)-1,-1,-1):
            if self.use_batchnorm:
                dx2, dW2, db2, dgamma2, dbeta2 = affine_norm_relu_backward(dx2, hidden_layer_cache[hidden_dim_idx])
                grads.update({'beta2_'+str(hidden_dim_idx): dbeta2, 'gamma2_'+str(hidden_dim_idx): dgamma2})
            else:
                dx2, dW2, db2 = affine_relu_backward(dx2, hidden_layer_cache[hidden_dim_idx])
            grads.update({'W2_'+str(hidden_dim_idx): dW2, 'b2_'+str(hidden_dim_idx): db2})
            dW2 += self.reg * self.params['W2_'+str(hidden_dim_idx)]

        print 'back prop into conv'
        dx2 = dx2.reshape(N, F, Hp, Wp)
        for conv_layer_num in range(self.num_of_conv_layers-1, -1, -1):
            if self.use_batchnorm:
                dx, dW, db, dgamma, dbeta = conv_norm_relu_pool_backward(dx2, cache_conv_layer[conv_layer_num])
                grads.update({'beta_'+str(conv_layer_num+1): dbeta, 'gamma_'+str(conv_layer_num+1): dgamma})
            else:
                dx, dW, db = conv_relu_pool_backward(dx2, cache_conv_layer[conv_layer_num])
            dW += self.reg * self.params['W_'+str(conv_layer_num+1)]
            grads.update({'W_'+str(conv_layer_num+1): dW, 'b_'+str(conv_layer_num+1): db})
            dx2 = dx
            print 'back prop into conv', conv_layer_num, ' finished'

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        return loss, grads
