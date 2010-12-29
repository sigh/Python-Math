from __future__ import division
from operator import mul
from primes import primes

"""
Factors
"""

def prime_factors(n):
    """Calculate prime factors of n

    Prime factors are returned in a dictionary with the primes factors
    as keys and the count of each factor as the values
    """
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

def factors(n):
    """Returns factors of n

    n can be a prime factor dictionary as returned by prime_factors(n)
    """

    if type(n) != dict: n = prime_factors(n)

    factors = [1]
    
    for (pf, count) in n.items():
        powers = [pf**i for i in range(1, count+1)]
        factors.extend([p*f for p in powers for f in factors])

    return factors

"""
Divisors function
"""

def num_divisiors(n):
    """Returns the number of divisors of n

    n can be a prime factor dictionary as returned by prime_factors(n)
    """
    if type(n) != dict: n = prime_factors(n)

    return reduce(mul, (a+1 for a in n.values()), 1)

def divisor_sum(n, x=1):
    """Returns the sum of divisors of n

    The divisors of will be raised to the power x.

    n can be a prime factor dictionary as returned by prime_factors(n)
    """
    if x == 0:
        return num_divisors(n)

    if type(n) != dict: n = prime_factors(n)

    result = 1
    for p,e in n.items():
        result *= (p**(x*(e+1)) - 1) // ( p**x - 1)
    return result

"""
Totient
"""

def totient(n):
    """Returns the number of positive integers (0 < i < n) co-prime to n

    n can be a prime factor dictionary as returned by prime_factors(n)
    """
    if type(n) == dict:
        return reduce(mul, ((p-1)*p**(e-1) for p,e in n.items()), 1)
    else:
        pfs = prime_factors(n)
        for p in pfs:
            n -= n//p
        return n

