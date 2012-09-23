from __future__ import division
from math import sqrt, floor

def rationals(a=(0,1)):
    """Enumerates the rationals in constant space but not in a nice order.

    This algorithm corresponds to a depth-first traversal of the
    Calkin-Wilf tree.

    See http://www.cs.ox.ac.uk/jeremy.gibbons/publications/rationals.pdf
    """
    while True:
        x, y = divmod(*a)
        a = (a[1], (a[1] * x) + (a[1] - y))
        yield a

def stern_bercot_constrained_gen(f=None, a=(0,1), b=(1,1)):
    """Generates the rationals in reduced form

    a and b can be used to set the bounds of the stern-bercot tree
    """
    stack = []

    if f == None:
        f = lambda x: True

    while True:
        c = a[0] + b[0] , a[1] + b[1]
        if f(*c):
            yield c
            stack.append((b,c))
            a,b = a,c
        elif stack:
            a,b = stack.pop()
        else:
            return

def farey_gen(n, start=(0,1), end=(1,1)):
    """Generates the nth row of the farey sequence

    start and end can be used if only a part of the row is required.
    """

    a, b = start
    x, y = end
    c, d = 1, n

    yield (a,b)
    yield (c,d)

    while ((c,d) != end):
        k = (n+b)//d
        e = k*c-a
        f = k*d-b
        a, b, c, d = c, d, e, f
        yield (c,d)

def gcd(a, b):
    """Returns the greatest common divisor of a and b

    If a and b are positive, result is always positive.
    Sign maybe positive or negative otherwise.
    """
    while b:
        a, b = b, a%b
    return a

def lcm(a,b):
    """Returns the lowest common multiple of a and b

    If a and b are positive, result is always positive.
    Sign maybe positive or negative otherwise.
    """
    return a*b//gcd(a,b)

def discrete_log(x, base, n):
    """Find y such that base**y = x (mod n)"""

    x = x%n
    base = base%n

    m = int(sqrt(n))+1

    T = {}
    b = 1
    for i in range(0,m):
        T[b] = i
        b = (b*base)%n

    c = modular_div(1,b,n)

    b=x
    j=0

    for j in range(m):
        if b in T:
            return (j*m+T[b])%n
        b = (b*c)%n

    return None

def best_approximations(m, n):
    """ Find the best approximations for m/n

    A best approximation is a rational a/b such that there does not exist
    a rational c/d which is closer to m/n with d <= b

    Assume 0 < m/n < 1
    """

    a, b = (0,1), (1,1)

    while 1:
        c = a[0] + b[0] , a[1] + b[1]

        yield c
        x, y = m*c[1], n*c[0]

        if x < y:
            a, b = a, c
        elif x > y:
            a, b = c, b
        else:
            return

def best_approximation(x, eps=10**-3):
    """ Find the best approximation for floating point value x

    A best approximation is a rational a/b such that there does not exist
    a rational c/d which is closer to x with d <= b

    Assume x > 0
    """

    # set up a to avoid iterating over integers
    a = (max(int(x)-1,0),1)
    # set up b to avoid iterating over 1/n values
    b = (1,max(int(1/x)-1,0))

    while True:
        c = a[0] + b[0] , a[1] + b[1]

        y, z = x*c[1], c[0]

        if abs(y-z) < eps:
            return c
        elif y < z:
            a, b = a, c
        else:
            a, b = c, b

def extended_gcd(a, b):
    """Return (r, s, d) where a*r + b*s = d and d = gcd(a,b)"""
    x,y = 0, 1
    lastx, lasty = 1, 0

    while b:
        a, (q, b) = b, divmod(a,b)
        x, lastx = lastx-q*x, x
        y, lasty = lasty-q*y, y

    return (lastx, lasty, a)

def modular_div(a, b, n):
    """Return a/d (mod n) assuming gcd(b,n) = 1"""
    return (a*(extended_gcd(b, n)[0]))%n

def chinese_remainder_theorem(items):
  """Solve the chinese remainder theorem

  Given a list of items (a_i, n_i) solve for x such that x = a_i (mod n_i)
  such that 0 <= x < product(n_i)

  Assumes that n_i are pairwise co-prime.
  """

  # Determine N, the product of all n_i
  N = 1
  for a, n in items:
    N *= n

  # Find the solution (mod N)
  result = 0
  for a, n in items:
    m = N//n
    r, s, d = extended_gcd(n, m)
    if d != 1:
      raise "Input not pairwise co-prime"
    result += a*s*m

  # Make sure we return the canonical solution.
  return result % N

def fp_continued_fraction(x,eps=10**-3):
    """Returns the finite continued fraction of the floating point number x

    The continued fraction [a0; a1 a2 ... an] of x is return in the form:
      (a0, [a1, a2, ..., an])

    eps may be used to set the desired accuracy
    """
    a0 = floor(x)
    x -= a0

    a_list = []

    while x > eps:
        x = 1.0/x
        a = floor(x)
        x -= a
        a_list.append(int(a))

    return (int(a0), a_list)

def rational_continued_fraction(p,q):
    """Returns the finite continued fraction of p/q

    The continued fraction [a0; a1 a2 ... an] of p/q is return in the form:
      (a0, [a1, a2, ..., an])
    """
    a0, rem = divmod(p,q)
    a_list = []

    while rem:
        p, q = q, rem
        a, rem = divmod(p,q)
        a_list.append(a)

    return (a0, a_list)

def sqrt_continued_fraction(n):
    """Calculates the infinite continued fraction of sqrt(n)

    The continued fraction [a0; a1 a2 ... an a1 a2 ... ] of sqrt(n) is
    return in the form:
      (a0, [a1, a2, ..., an])
    """
    sqrt_n = sqrt(n)

    int_sqrt_n = int(sqrt_n)
    if int_sqrt_n**2 == n:
        return (int_sqrt_n,[])

    init_remainder = (1,int_sqrt_n)
    a_list = []

    remainder = init_remainder
    while 1:
        x,y = remainder

        next_a = int(x/(sqrt_n-y))
        new_x = (n-y**2)/x
        new_y = -y + new_x*next_a
        remainder = (new_x, new_y)

        a_list.append(next_a)

        if remainder == init_remainder:
            return (int_sqrt_n,a_list)

def generalised_ppt_gen(f, init=None):
    """Generate primitive solutions to a**2 + b**2 = c**2

    Generate primitive pythagorean triples by the method given in
     http://mathworld.wolfram.com/PythagoreanTriple.html
     references Roberts, J "Elementary Number Theory: A problem Oriented Approach" (1977)

    Searches the tree given by the recurrance until f(a,b,c) is False where

    Given the approporate inital conditions, this recurrance solves for any
    equation of the form a**2 + b**2 = c**2 + k
    """

    if init == None:
        stack = [(3,4,5)]
    else:
        stack = list(init)

    a,b,c = stack.pop()

    while True:
        if f(a,b,c):
            yield (a,b,c)

            a2, b2, c2, c3 = a*2, b*2, c*2, c*3

            stack.append((a+b2+c2,a2+b+c2,a2+b2+c3))
            if a != b:
                stack.append((a-b2+c2,a2-b+c2,a2-b2+c3))

            a,b,c = -a+b2+c2, -a2+b+c2, -a2+b2+c3

        elif stack:
            a,b,c = stack.pop()
        else:
            return

def pythagorean_triple_gen(f):
    """Generates pythagorean triplets

    Find all (a,b,c) such that a**2 + b**2 = c**2
    where a,b > 0 and c > 2

    Branches are constrained by the boolean function f(a,b,c)
    """
    for x0,y0,z0 in generalised_ppt_gen(f):
        x,y,z = x0,y0,z0

        while f(x,y,z):

            yield (x,y,z)
            x += x0
            y += y0
            z += z0
