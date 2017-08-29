import numpy as np
import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1-1e-7):
    updates = []
    all_grads = theano.grad(loss, all_params)

    i = theano.shared(np.float32(1))  # HOW to init scalar shared?
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED
    learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)

    for param_i, g in zip(all_params, all_grads):
        m = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t TO use beta1_t
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t) )
    updates.append((i, i_t))
    return updates

def sgd(loss, all_params, learning_rate=0.0002):

    updates = []
    all_grads = theano.grad(loss, all_params)
    learning_rate_t = learning_rate

    for param_i, g in zip(all_params, all_grads):
        param_i_t = param_i - (learning_rate_t * g)
        updates.append((param_i, param_i_t) )
    return updates

def RMSprop(cost, params, learning_rate=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates
