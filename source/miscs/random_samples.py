import numpy as np
import chainer
from scipy.stats import truncnorm


def sample_continuous(dim, batchsize, distribution='normal', xp=np, trunc_flag=False, threshold=1):
    if distribution == "normal":
        if trunc_flag:
            # Can use the truncation trick to sample only between the specified threshold.
            z = truncnorm(-threshold, threshold).rvs(size=(batchsize, dim), random_state=0)
            print(z.astype(xp.float32))
            print(xp.random.randn(batchsize, dim) \
            .astype(xp.float32))
            exit()
            return z.astype(xp.float32)
        else:
            # Otherwise sample from entire distribution
            # randn samples from a normal distribution of mean=0 and variance = 1
            return xp.random.randn(batchsize, dim) \
            .astype(xp.float32)
    elif distribution == "uniform":
        return xp.random.uniform(-1, 1, (batchsize, dim)) \
            .astype(xp.float32)
    else:
        raise NotImplementedError


def sample_categorical(n_cat, batchsize, distribution='uniform', xp=np):
    if distribution == 'uniform':
        return xp.random.randint(low=0, high=n_cat, size=(batchsize)).astype(xp.int32)
    else:
        raise NotImplementedError


def sample_from_categorical_distribution(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.
    Args:
        batch_probs (ndarray): batch of action probabilities BxA
    Returns:
        ndarray consisting of sampled action indices
    """
    xp = chainer.cuda.get_array_module(batch_probs)
    return xp.argmax(
        xp.log(batch_probs) + xp.random.gumbel(size=batch_probs.shape),
        axis=1).astype(np.int32, copy=False)
