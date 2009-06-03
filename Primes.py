from operator import mul
from heapq import heappush, heapreplace
from itertools import chain, takewhile
from math import sqrt
from bisect import bisect_left

from Singleton import Singleton

class Primes(Singleton):
    SMALL_PRIME_LIMIT = 50
    SMALL_PRIME_LIMIT_SQR = SMALL_PRIME_LIMIT**2
    
    PRIME_LIMIT = 2147483647

    _SQRT_LIMIT = int(sqrt(PRIME_LIMIT))
    _INITIAL_PRIMES = [2, 3, 5, 7]
            
    def __new__(cls, *args, **kwargs):
        self = Singleton.__new__(cls, *args, **kwargs)

        if '_primes' not in vars(self):
            self._primes = []
            self._lookup = {}
            self._count = 0
            self._max_prime = 1
            self._iter = self._prime_gen(self._INITIAL_PRIMES)

            # create initial list of primes
            g = self._continue_iter()
            while g.next() < self.SMALL_PRIME_LIMIT:
                pass
            
            self.SMALL_PRIMES = list(self._primes)
            
        return self

    def __contains__(self, n):
        return self.is_prime(n)

    def __getitem__(self, n):
        g = self._continue_iter()
        
        while self._count <= n:
            g.next()
            
        return self._primes[n]
        
    def __iter__(self):
        return chain(iter(self._primes), self._continue_iter())

    """
    Range
    """

    def __getslice__(self, start, end):
        if self._count < end:
            it = self._continue_iter()
            while self._count < end:
                it.next()
            
        return self._primes[start:end]

    def upto(self, limit):
        return takewhile(lambda x: x < limit+1, self)

    """
    Is prime
    """

    def is_prime(self, n):
        if n < primes._max_prime:
            return (n in primes._lookup)
        if any( n%p == 0 for p in primes.SMALL_PRIMES ):
            return False
        if n < primes.SMALL_PRIME_LIMIT_SQR:
            return True
        
        return miller_rabin_safe(n)


    def next_prime(self, n):
        if n > self._max_prime:
            while 1:
                if self.is_prime(n):
                    return n
                n += 1
        else:
            return self._primes[bisect_left(self._primes, n)]

    """
    Prime number generator
    """
    
    # continue running the internal iterator
    # and processing the resulting primes
    def _continue_iter(self, upto = PRIME_LIMIT):
        for p in self._iter:
            if p >= upto:
                return
            
            self._primes.append(p)
            self._count += 1
            self._lookup[p] = self._count
            self._max_prime = p
            yield p

    def _prime_gen(self, initial_primes):
        for p in initial_primes:
            yield p
        
        def insert(p):
            if p < self._SQRT_LIMIT:
                heappush(heap, (p*p, 2*p))

        wheel = self._prime_wheel(initial_primes)
        
        heap = []
        c = wheel.next()
        insert(c)
        yield c
        
        for c in wheel:
            n, i = heap[0]
            while n < c: 
                heapreplace(heap, (n+i, i))
                n, i = heap[0]

            if n > c:
                insert(c)
                yield c
                continue
            
    def _prime_wheel(self, initial_primes):
        limit = reduce(mul, initial_primes)

        candidates = []

        # find the candidate primes which aren't divisible by any of
        # the given primes
        for n in range(2, limit+2):
            for p in initial_primes:
                if n%p == 0:
                    break
            else:
                candidates.append(n)

        num_candidates = len(candidates)

        while 1:
            for i in range(num_candidates):
                yield candidates[i]
                candidates[i] += limit

primes = Primes()

"""
Is prime
"""

is_prime = primes.is_prime
next_prime = primes.next_prime

"""
Miller Rabin
"""

def miller_rabin_safe(n):
    if n < 1373653:
         return miller_rabin(n,[2, 3])
    if n < 25326001:
        return miller_rabin( n, [2, 3, 5] )
    if n < 2152302898747:
        return miller_rabin( n, [2, 3, 5, 7, 11] )
    if n < 3474749660383:
        return miller_rabin( n, [2, 3, 5, 7, 11, 13] )
    if n < 341550071728321:
        return miller_rabin( n, [2, 3, 5, 7, 11, 13, 17] )
    if n < 10000000000000000:
        return miller_rabin( n, [2, 3, 7, 61, 24251] )
    else:
        return miller_rabin( n, [2,3,5,7,11,13,19] )

def miller_rabin(n, bases):
    s = 0
    t = n-1
    while t & 1 == 0:
        t >>= 1
        s += 1

    return all(_miller_rabin_test(n,b, s, t) for b in bases)

def _miller_rabin_test(n, base, s, t):
    if n < 2:
        return False

    b = pow(base, t, n)
    
    if b == 1 or b == n-1:
        return True

    for j in range(1, s):
        b = (b*b) % n
        if b == n-1:
            return True

    return False

"""
Factors
"""
def factors(n):
    if type(n) == dict:
        pfs = n
    else:
        pfs = prime_factors(n)
        
    factors = [1]
    
    for (pf, count) in pfs.items():
        powers = [pf**i for i in range(1, count+1)]
        factors.extend([p*f for p in powers for f in factors])

    return factors

def prime_factors(n):
    pf = {}
    
    for p in primes:
        if p*p > n:
            break

        count = 0
        while n%p == 0:
            n /= p
            count += 1
            
        if count:
            pf[p] = count

    if n > 1:
        pf[n] = 1

    return pf


"""
Totient
"""

def totient(n):
    if type(n) == dict:
        return reduce(mul, ((p-1)*p**(e-1) for (p,e) in n.items()))
    else:
        pfs = prime_factors(n)
        for p in pfs:
            n -= n/p
        return n

