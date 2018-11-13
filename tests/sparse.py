
import numpy as np

from opymize.linear.sparse import transposeopn, einsumop, diffopn

def test_transposeop():
    N = (7,5,6,3)
    axes = [(3,1,0,2), (0,1,2,3), (0,1,3,2), (1,0,2,3)]
    A = np.random.randn(*N)
    for a in axes:
        AT = A.transpose(*a).ravel()
        assert np.linalg.norm(transposeopn(N, a).dot(A.ravel()) - AT) == 0.0

einsum_tests = [
    [       ",->"  ,      None,        (),      (),    None,    (),   ()],
    [      "i,->"  ,      None,      (3,),      (),    None,    (),   ()],
    [      ",i->"  ,      None,        (),    (6,),    None,    (),   ()],
    [     "i,i->"  ,      None,      (5,),    (5,),    None,    (),   ()],
    [      "i,->i" ,      None,      (5,),      (),    None,  (5,),   ()],
    [     ",ij->ji",      None,        (),   (5,6),    None, (6,5),   ()],
    [    "i,ij->j" ,      None,      (5,),   (5,6),    None,  (6,),   ()],
    [  "jkl,jl->jk",      None,   (5,3,4),    None,   (5,4), (5,3),   ()],
    [  "jkl,lj->jk",      None,   (5,3,4),   (4,5),    None, (5,3),   ()],
    [  "jkl,jl->kj",      None,   (5,3,4),   (5,4),    None, (3,5),   ()],
    [  "jkl,lj->kj",      None,   (5,3,4),   (4,5),    None, (3,5),   ()],
    [  "jkl,lj->kj",      None,   (5,3,4),   (4,5),    None,  None,   ()],
    ["jmk,jlm->jlk",      None,   (5,4,6), (5,3,4),    None,  None,   ()],
    ["ijkl,kij->lj",      None, (4,3,6,5),    None, (6,4,3),  None,   ()],
    [ "ij,kjl->kil",      None,     (5,3), (4,3,2),    None,  None,   ()],
    [       ",->i" ,    ",->" ,        (),      (),    None,  (7,), (0,)],
    [     "j,j->ji",  "j,j->j",      (5,),    (5,),    None, (5,2), (1,)],
    [    "jk,k->ji", "jk,k->j",     (5,4),    (4,),    None, (5,2), (1,)],
]

def test_einsumop(d1, d2, sop, sin1, sin2, sout, rd):
    sin2 = sin1 if sin2 is None else sin2
    for order in ['C', 'F', ('C','F'), ('F','C')]:
        if order in ['C','F']:
            order_in = order_out = order
        else:
            order_in, order_out = order

        data = np.random.randn(*sop)
        x = np.asarray(np.random.randn(*sin2))

        d2 = d1 if d2 is None else d2
        y = np.asarray(np.einsum(d2, data, x))
        if len(rd) > 0:
            sl = [None if j in rd else slice(None) for j in range(len(sout))]
            y = y[tuple(sl)]

        A = einsumop(d1, data, shape_in=sin1, shape_out=sout, order=order)
        sout = y.shape if sout is None else sout
        Ax = A.dot(x.ravel(order=order_in)).reshape(sout, order=order_out)

        assert np.linalg.norm((y - Ax).ravel(), ord=np.inf) < 1e-12


test_v = np.array([[0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0.]])
test_Dv = {
    "centered": np.array([[[ 0.5,  0.5,  0. ,  0. ,  0. ],
                           [-0.5, -0.5,  0. ,  0.5,  0. ],
                           [ 0. ,  0. ,  0. , -0.5,  0. ],
                           [ 0. ,  0. ,  0. ,  0. ,  0. ]],
                          [[ 0.5, -0.5,  0. ,  0. ,  0. ],
                           [ 0.5, -0.5,  0. ,  0.5,  0. ],
                           [ 0. ,  0. ,  0. ,  0.5,  0. ],
                           [ 0. ,  0. ,  0. ,  0. ,  0. ]]]),
    "central": np.array([[[ 0. ,  0. ,  0. ,  0. ,  0. ],
                          [ 0. ,  0. ,  0. ,  0. ,  0.5],
                          [ 0. , -0.5,  0. ,  0. ,  0. ],
                          [ 0. ,  0. ,  0. ,  0. ,  0. ]],
                         [[ 0. ,  0. ,  0. ,  0. ,  0. ],
                          [ 0. ,  0. , -0.5,  0. ,  0. ],
                          [ 0. ,  0. ,  0. ,  0.5,  0. ],
                          [ 0. ,  0. ,  0. ,  0. ,  0. ]]]),
}

def test_gradop(s, c):
    cc = min(1,c-1)

    Dv = test_Dv[s].transpose(1,2,0).ravel()
    N = test_v.shape
    v = test_v.ravel()

    w = np.random.randn(c*np.prod(N)).astype(np.float64)
    w.reshape(-1,c)[:,cc] = v

    Dw = diffopn(N, components=c, schemes=s, boundaries="neumann").dot(w)
    assert np.linalg.norm(Dw.reshape(-1,c)[:,cc] - Dv, ord=np.inf) == 0.0

if __name__ == "__main__":
    print("=> Transpose operations")
    test_transposeop()

    print("=> Sparse matrix representations of dense einsum operations:")
    for i, t in enumerate(einsum_tests):
        print("#%02d (%s)" % (i, t[0]))
        test_einsumop(*t)

    for s in ["centered", "central"]:
        for c in [1,2,3]:
            print("=> Gradient with %s scheme and %d channels" % (s,c))
            test_gradop(s, c)
