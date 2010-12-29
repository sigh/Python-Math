from __furture__ import division
import operator

def subset_gen(xs):
    """ Generates all subsets of xs
    
    Note: The same list is used for each permutation, thus a copy
    should be made if persistence is required.
    """
    def generator(n,ys):
        if n == 0:
            yield ys
        else:
            n -= 1
            
            for res in generator(n, ys):
                yield res

            ys.append(x[n])
            
            for res in generator(n, ys):
                yield res

            ys.pop()
            
    return generator(len(xs),[])

def combination_gen(xs, k):
    """ Generates all combinations of k items in xs
    
    Note: The same list is used for each permutation, thus a copy
    should be made if persistence is required.
    """
    def generator(n,ys,k):
        if k == 0:
            # found a subset of the correct length
            yield ys
        elif n == k:
            # all remaining items must be included in the subset
            while k:
                k -= 1
                n -= 1
                ys[k] = xs[n]
            yield ys
        else:
            n -= 1

            # exclude current item
            for res in generator(n, ys, k):
                yield res

            k -= 1
            ys[k] = xs[n]

            # include current item
            for res in generator(n, ys, k):
                yield res
                
    k = max(min(k,len(xs)),0)
    return generator(len(xs),[0]*k,k)

def permutation_gen(xs):
    """ Generates all permutations of the elements in xs.

    Permutations are in lexographic order, assuming that
    xs is sorted initially.
    Each item in xs is treated as unique.
    
    Note: The same list is used for each permutation, thus a copy
    should be made if persistence is required.
    """
    
    n = len(xs)
    index = range(-1, n)

    yield xs
    
    while 1:
        k = n-1
        while index[k] > index[k+1]:
            k -= 1

        if k == 0:
            return

        j = n
        while index[k] > index[j]:
            j -= 1

        index[k], index[j] = index[j], index[k]
        xs[k-1], xs[j-1] = xs[j-1], xs[k-1]
        
        r = n
        s = k+1
        while r > s:
            index[r], index[s] = index[s], index[r]
            xs[r-1], xs[s-1] = xs[s-1], xs[r-1]
            r -= 1
            s += 1

        yield xs

def factorial(n):
    """Returns n!
    
    n is truncated to an integer
    values of n < 0 return 0.
    """
    if n < 0:
        return 0
    
    return reduce(operator.mul, range(1,n+1), 1)
    
def nCr(n, r):
    """Return the number of ways of choosing r out of n items"""
    a = 1
    b = 1

    if r > n//2:
        r = n-r
    
    while r > 0:
        a *= n
        b *= r
        n -= 1
        r -= 1

    return a//b
