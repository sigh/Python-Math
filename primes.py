from operator import mul
from heapq import heappush, heapreplace
from itertools import chain, takewhile
from math import sqrt
from bisect import bisect_left

class Primes(object):
    """This class acts as an infinite prime list
    """

    SMALL_PRIME_LIMIT = 50
    SMALL_PRIME_LIMIT_SQR = SMALL_PRIME_LIMIT**2
    
    _INITIAL_PRIMES = [2, 3, 5, 7]
            
    def __new__(cls, *args, **kwargs):
        # ensure primes is a singleton
        if '_inst' not in vars(cls):
            cls._inst = object.__new__(cls, *args, **kwargs)
        self = cls._inst

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
        """Test if n is a prime number"""
        return self._is_prime(n)

    def __getitem__(self, n):
        """Return the nth prime"""
        g = self._continue_iter()
        
        while self._count <= n:
            g.next()
            
        return self._primes[n]
        
    def __iter__(self):
        """Return an iterator over all the primes"""
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
        """Return primes upto a given limit"""
        return takewhile(lambda x: x < limit+1, self)

    """
    Is prime
    """

    def _is_prime(self, n):
        """Test if n is a prime number"""

        # if we've generated primes greater than n we can
        # just check our lookup table
        if n <= primes._max_prime:
            return (n in primes._lookup)
        
        # otherwise use miller-rabin
        return miller_rabin(n)


    def _next_prime(self, n):
        """Return the smallest prime greater or equal to n"""
        if n > self._max_prime:
            while 1:
                if self._is_prime(n):
                    return n
                n += 1
        else:
            return self._primes[bisect_left(self._primes, n)]

    """
    Prime number generator
    """
    
    # continue running the internal iterator
    # and processing the resulting primes
    def _continue_iter(self):
        """Continue running the internal iterator"""
        for p in self._iter:
            self._primes.append(p)
            self._count += 1
            self._lookup[p] = self._count
            self._max_prime = p
            yield p

    def _prime_gen(self, initial_primes):
        """An infinite generator for prime numbers"""

        # first output the primes that the generator
        # is seed with
        for p in initial_primes:
            yield p
        
        wheel = self._prime_wheel(initial_primes)
        
        # This heap stores composite numbers. Items are of the form:
        # (n,i)
        # if i < 0 then i is the index of the prime p such that p*p = n
        # if i > 0 then i is how much n should be incremented when
        #          reinserting into the heap
        heap = []

        # output the first prime after the initial primes
        p = wheel.next()
        yield p
        initial_index = len(initial_primes)
        heappush(heap, (p*p, -initial_index))
        
        for c in wheel:
            n, i = heap[0]
            while n < c: 
                if i < 0:
                    # reinsert in (n, increment) format
                    p2 = 2*self._primes[-i]
                    heapreplace(heap, (n+p2, p2))

                    # increment the index
                    i -= 1
                    p = self._primes[-i]
                    heappush(heap, (p*p, i))
                else:
                    # increment n
                    heapreplace(heap, (n+i, i))

                n, i = heap[0]

            if n > c:
                yield c
            
    def _prime_wheel(self, initial_primes):
        """Generate candidates to test for primality
        
        Generates an inifite list of numbers swhich aren't divisible by any 
        of the given initial primes
        """

        limit = reduce(mul, initial_primes)

        candidates = []

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

def is_prime(p):
    """Tests if p is prime"""
    return p in primes

next_prime = primes._next_prime

"""
Miller Rabin
"""

def miller_rabin(n, bases=None):
    """Tests if n is prime using miller-rabin
    
    Uses the given list of bases as witnesses. If no bases
    are provided, a good set will be chosen
    """

    if bases == None:
        bases = _miller_rabin_bases(n)

    s = 0
    t = n-1
    while t & 1 == 0:
        t >>= 1
        s += 1

    return all(_miller_rabin_test(n,b, s, t) for b in bases)

def _miller_rabin_test(n, base, s, t):
    """Return True if n is probably prime"""
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

def _miller_rabin_bases(n):
    """Choose default bases for miller-rabin
    
    These bases are sufficient to test for any number up to 341,550,071,728,321
    http://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
    """
    if n < 1373653:
         return [2, 3]
    if n < 9080191:
         return [31, 73]
    if n < 4759123141:
        return [2, 7, 61]
    if n < 2152302898747:
        return [2, 3, 5, 7, 11]
    if n < 3474749660383:
        return [2, 3, 5, 7, 11, 13]
    if n < 341550071728321:
        return [2, 3, 5, 7, 11, 13, 17]
    if n < 10000000000000000:
        return [2, 3, 7, 61, 24251]
    else:
        return [2,3,5,7,11,13,19]
