
import numpy as np

from opymize.linear.sparse import transposeopn, einsumop, diffopn

print("=> Transpose operations")
N = (7,5,6,3)
axes = [(3,1,0,2), (0,1,2,3), (0,1,3,2), (1,0,2,3)]
A = np.random.randn(*N)
for a in axes:
    AT = A.transpose(*a).ravel()
    assert np.linalg.norm(transposeopn(N, a).dot(A.ravel()) - AT) == 0.0

print("=> Sparse matrix representations of dense einsum operations:")
tests = [
    [       ",->"  ,      None,        (),      (),    (),   ()],
    [      "i,->"  ,      None,      (3,),      (),    (),   ()],
    [      ",i->"  ,      None,        (),    (6,),    (),   ()],
    [     "i,i->"  ,      None,      (5,),    (5,),    (),   ()],
    [      "i,->i" ,      None,      (5,),      (),  (5,),   ()],
    [     ",ij->ji",      None,        (),   (5,6), (6,5),   ()],
    [    "i,ij->j" ,      None,      (5,),   (5,6),  (6,),   ()],
    [  "jkl,jl->jk",      None,   (5,3,4),   (5,4), (5,3),   ()],
    [  "jkl,lj->jk",      None,   (5,3,4),   (4,5), (5,3),   ()],
    [  "jkl,jl->kj",      None,   (5,3,4),   (5,4), (3,5),   ()],
    [  "jkl,lj->kj",      None,   (5,3,4),   (4,5), (3,5),   ()],
    [  "jkl,lj->kj",      None,   (5,3,4),   (4,5),  None,   ()],
    ["ijkl,kij->lj",      None, (4,3,6,5), (6,4,3),  None,   ()],
    [       ",->i" ,    ",->" ,        (),      (),  (7,), (0,)],
    [     "j,j->ji",  "j,j->j",      (5,),    (5,), (5,2), (1,)],
    [    "jk,k->ji", "jk,k->j",     (5,4),    (4,), (5,2), (1,)],
]
for i, (d1, d2, sop, sin, sout, rd) in enumerate(tests):
    print("#%02d (%s)" % (i, d1))

    data = np.random.randn(*sop)
    x = np.asarray(np.random.randn(*sin))

    d2 = d1 if d2 is None else d2
    y = np.asarray(np.einsum(d2, data, x))
    if len(rd) > 0:
        sl = [None if j in rd else slice(None) for j in range(len(sout))]
        y = y[tuple(sl)]

    A = einsumop(d1, data, sin, shape_out=sout)
    sout = y.shape if sout is None else sout
    Ax = A.dot(x.ravel()).reshape(sout)

    assert np.linalg.norm((y - Ax).ravel(), ord=np.inf) < 1e-12

print("=> Gradient with centered scheme and Neumann boundaries:")
N = (4,5)
v = np.random.randint(low=0, high=10, size=np.prod(N)).astype(np.float64)
print(v.reshape(N))
D = diffopn(N, schemes="centered")
print(D.dot(v).reshape((np.prod(N), len(N))).T.reshape((len(N),) + N))

print("=> Gradient with central scheme and Neumann boundaries:")
N = (4,5)
v = np.random.randint(low=0, high=10, size=3*np.prod(N)).astype(np.float64)
print(v.reshape(N + (3,))[:,:,1])
D = diffopn(N, components=3, schemes="central", boundaries="neumann")
print(D.dot(v).reshape(N + (len(N),3)).transpose(2,0,1,3)[:,:,:,1])
