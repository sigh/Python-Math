from __future__ import division
from operator import mul
from my.primes import primes

"""
Factors
"""
def prime_factors(n):
    pf = {}
    
    for p in primes:
        if p*p > n:
            break
        
        if n%p == 0:
            count = 1
            n //= p

            while n%p == 0:
                n //= p
                count += 1

            pf[p] = count

    if n > 1:
        pf[n] = 1

    return pf

def factors(pfd):
    if type(pfd) != dict: pfd = prime_factors(pfd)

    factors = [1]
    
    for (pf, count) in pfd.items():
        powers = [pf**i for i in range(1, count+1)]
        factors.extend([p*f for p in powers for f in factors])

    return factors

"""
Divisors function
"""

def num_divisiors(pfd):
    if type(pfd) != dict: pfd = prime_factors(pfd)

    return reduce(mul, (a+1 for a in pfd.values()), 1)

def divisor_sum(pfd, x=1):
    if x == 0:
        return num_divisors(pfd)

    if type(pfd) != dict: pfd = prime_factors(pfd)

    result = 1
    for p,e in pfd.items():
        result *= (p**(x*(e+1)) - 1) // ( p**x - 1)
    return result

"""
Totient
"""

def totient(n):
    if type(n) == dict:
        return reduce(mul, ((p-1)*p**(e-1) for p,e in n.items()), 1)
    else:
        pfs = prime_factors(n)
        for p in pfs:
            n -= n//p
        return n

