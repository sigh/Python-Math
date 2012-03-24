from operator import mul
from heapq import heappush, heapreplace
from itertools import chain, takewhile, count
from math import sqrt
from bisect import bisect_left

class Primes(object):
    """Infinite prime list (singleton)

    Can do everything you would expect from an infinite list:
    - retrieve primes by index
    - slice the list
    - iterate over the list
    - test if the list contains a given value

    >>> 5 in primes
    True
    >>> 51 in primes
    False
    >>> primes[99] # 100th prime
    541
    >>> primes[5:10]
    [13, 17, 19, 23, 29]
    >>> list(primes.upto(50))
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    """

    SMALL_PRIME_LIMIT = 50

    _INITIAL_PRIMES = [2, 3, 5, 7]

    def __new__(cls, *args, **kwargs):
        # ensure this class is a singleton
        if '_inst' not in vars(cls):
            cls._inst = object.__new__(cls, *args, **kwargs)
        return cls._inst

    def __init__(self):
        self._iter = self._internal_iterator()

        # create initial list of primes
        while self._iter.next() < self.SMALL_PRIME_LIMIT:
            pass

    def __contains__(self, n):
        """Test if n is a prime number"""
        return self._is_prime(n)

    def __getitem__(self, n):
        """Return the (n+1)th prime"""

        while self._count <= n:
            self._iter.next()

        return self._primes[n]

    def __iter__(self):
        """Return an iterator over all the primes"""

        for n in count():
            if self._count == n:
                # if count is EQUAL to n then we want the very next prime
                yield self._iter.next()
            else:
                # if count is GREATER than n then we have already generated this prime
                yield self._primes[n]
            # count can never be LESS than n (because we must have generated all primes
            # up to the nth prime already)

    """
    Range
    """

    def __getslice__(self, start, end):
        while self._count < end:
            self._iter.next()

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
            while True:
                if self._is_prime(n):
                    return n
                n += 1
        else:
            return self._primes[bisect_left(self._primes, n)]

    """
    Prime number generator
    """

    def _internal_iterator(self):
        """Generate and store primes

        Note: There should only be ONE of this iterator active per instance
        """

        if '_primes' in vars(self):
            raise Exception("Primes._internal_iterator should only be called ONCE per instance")

        self._primes = []
        self._lookup = {}
        self._count = 0
        self._max_prime = 1

        for p in self._prime_gen(self._INITIAL_PRIMES):
            self._primes.append(p)
            self._count += 1
            self._lookup[p] = self._count
            self._max_prime = p
            yield p

    def _prime_gen(self, initial_primes):
        """An infinite generator for prime numbers"""

        # first output the primes that the generator is seeded with
        for p in initial_primes:
            yield p

        wheel = self._prime_wheel(initial_primes)

        # output the first prime after the initial primes
        p = wheel.next()
        yield p

        # This heap stores composite numbers. Items are of the form:
        # (n,i)
        # if i < 0 then i is the index of the prime p such that p*p = n
        # if i > 0 then i is how much n should be incremented when
        #          reinserting into the heap.
        #
        # We could avoid the i < 0 case by inserting (p*p,2*p) for every prime
        #   after the intial primes but that wastes a lot more space
        heap = []

        initial_index = len(initial_primes)
        heappush(heap, (p*p, -initial_index))

        for c in wheel:
            n, i = heap[0]
            while n < c:
                if i < 0:
                    # reinsert in (n, increment) format
                    # the incement is TWICE the prime since even multiple
                    #   do not need to be checked
                    p2 = 2*self._primes[-i]
                    heapreplace(heap, (n+p2, p2))

                    # increment the index:
                    #   insert (p*p, -index) for the next highest prime
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

        Generates an infinite list of numbers swhich aren't divisible by any
        of the given initial primes
        """

        limit = reduce(mul, initial_primes)

        candidates = []

        # find all values 2 < c <= limit+1 such that c is not divisible
        #   by any of the initial primes
        for n in range(2, limit+2):
            for p in initial_primes:
                if n%p == 0:
                    break
            else:
                candidates.append(n)

        num_candidates = len(candidates)

        # for each c yield all values of c+limit*n
        #   this values will be guarenteed to contain
        #   all primes except for the initial_primes
        # (of course, many composite numbers will be included as well)
        while True:
            for i in range(num_candidates):
                yield candidates[i]
                candidates[i] += limit

primes = Primes()

def is_prime(p):
    """Tests if p is prime"""
    return p in primes

def next_prime(n):
    """Return the smallest prime greater or equal to n"""
    return primes._next_prime(n)

"""
Miller Rabin
"""

def miller_rabin(n, bases=None):
    """Tests if n is prime using miller-rabin

    Uses the given list of bases as witnesses. If no bases
    are provided, a good set will be chosen

    Note: only use for n > 3
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
