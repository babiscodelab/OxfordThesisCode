#!/usr/bin/env python
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(__file__))
from Sobol.digitalseq_b2g import digitalseq_b2g
from numpy.random import randint
from numpy import vstack, array

def sobol(kstart=0, m=None, s=None, scramble = True, deepcopy=True):
    # generate 2**m points in s dimensions

    # straightforward enumeration of Sobol points
    # using generating matrices from file "sobol_Cs.col":
    gen = digitalseq_b2g(os.path.join(os.path.realpath(__file__), "..", "sobol_Cs.col"), kstart=kstart, m=m, s=s, returnDeepCopy=deepcopy)

    if scramble == False:
        return vstack([x for x in gen])
    else:
        t = gen.t # this is the power of 2 at which the integers are shifted inside the generator
        t = max(32, t) # we will guarantee at least a depth of 32 bits for the shift
        ct = max(0, t - gen.t) # this is the correction factor to scale the integers
        # now using digital shifting
        shift = randint(2**t, size=(1,s), dtype=np.int64) # generate random shift
        # now generate the points and shift them
        return (shift ^ (vstack([array(gen.cur) for x in gen]) * 2**ct)) / 2.**t

def scramble(seq, t = 32):
    # seq is the unscrambled sequence. t must be the power of 2 at which the integers are shifted inside the generator
    s = seq.shape[1]
    t = max(32, t) # we will guarantee at least a depth of 32 bits for the shift
    seq = (seq * 2**t).astype('int64')
    shift = randint(2**t, size=(1,s), dtype=np.int64) # generate random shift
    return (shift ^ seq)/2.**t

if __name__ == '__main__':

    import numpy.random as npr

    m = 3
    s = 2

    unscrambled = sobol(m=m, s=s, scramble=False)
    print(unscrambled)

    npr.seed(123456789)
    print(sobol(m=m, s=s))

    npr.seed(123456789)
    print(scramble(unscrambled))
