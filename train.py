# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Training.
'''

from prepro import Hyperparams, load_data, load_charmaps
import sugartensor as tf

def get_batch_data():
    '''Makes batch queues from the data.

    Returns:
      A Tuple of x (Tensor), y (Tensor).
      x and y have the shape [batch_size, maxlen].
    '''
    # Load data
    X, Y = load_data()
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32), 
                                                  tf.convert_to_tensor(Y, tf.int32)])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=Hyperparams.batch_size, 
                                  capacity=Hyperparams.batch_size*64,
                                  min_after_dequeue=Hyperparams.batch_size*32, 
                                  allow_smaller_final_batch=False) 
    
    return x, y # (16, 100), (16, 100)

# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate
    opt += tf.sg_opt(size=3, rate=1, causal=False)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    # reduce dimension
    input_ = (tensor
              .sg_bypass(act='relu', bn=(not opt.causal), ln=opt.causal)
              .sg_conv1d(size=1, dim=in_dim/2, act='relu', bn=(not opt.causal), ln=opt.causal))

    # 1xk conv dilated
    out = input_.sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, act='relu', bn=(not opt.causal), ln=opt.causal)

    # dimension recover and residual connection
    out = out.sg_conv1d(size=1, dim=in_dim) + tensor

    return out

# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, is_train=True):
        '''
        Args:
          is_train: Boolean. If True, backprop is executed.
        '''
        if is_train:
            self.x, self.y = get_batch_data() # (16, 100), (16, 100)
        else:
#             self.x = tf.placeholder(tf.int32, [Hyperparams.batch_size, Hyperparams.maxlen])
            self.x = tf.placeholder(tf.int32, [None, Hyperparams.maxlen])
        
        # make embedding matrix for input characters
        hangul2idx, _, hanja2idx, _ = load_charmaps()
        
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(hangul2idx), dim=Hyperparams.hidden_dim)
        
        # embed table lookup
        self.enc = self.x.sg_lookup(emb=self.emb_x).sg_float() # (16, 100, 200)
        
        # loop dilated conv block
        for i in range(2):
            self.enc = (self.enc
                   .sg_res_block(size=5, rate=1)
                   .sg_res_block(size=5, rate=2)
                   .sg_res_block(size=5, rate=4)
                   .sg_res_block(size=5, rate=8)
                   .sg_res_block(size=5, rate=16))
        
        # final fully convolutional layer for softmax
        self.logits = self.enc.sg_conv1d(size=1, dim=len(hanja2idx)) # (16, 100, 4543)
        
        if is_train:
            self.ce = self.logits.sg_ce(target=self.y, mask=True) # (16, 100)
            self.nonzeros = tf.not_equal(self.y, tf.zeros_like(self.y)).sg_float() # (16, 100)
            self.reduced_loss = self.ce.sg_sum() / self.nonzeros.sg_sum() # ()
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")

def train():
    g = ModelGraph()
    print "Graph loaded!"
    tf.sg_train(lr=0.0001, lr_reset=True, log_interval=10, loss=g.reduced_loss, eval_metric=[], max_ep=20, 
                save_dir='asset/train', early_stop=False, max_keep=5)
     
if __name__ == '__main__':
    train(); print "Done"
