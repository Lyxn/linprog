from lu.dense import *


def mock_mat0(n, alpha=0.0):
    return np.resize(np.arange(n * n), (n, n)) + np.eye(n) * alpha


def get_lower(lu):
    return np.tril(lu, -1) + np.eye(len(lu))


def get_upper(lu):
    return np.triu(lu)


def test_lu0():
    n = 100
    mat = mock_mat0(n, n / 2)
    m0 = np.copy(mat)
    print("Matrix\n%s" % mat)
    lu, p, q = compute_lu(mat, func_piv=pivot_max_root)
    rows = piv2idx(p)
    cols = piv2idx(q)
    print("P\n%s\n%s" % (p, rows))
    print("Q\n%s\n%s" % (q, cols))
    # print("LU\n%s" % lu)
    lo = get_lower(lu)
    up = get_upper(lu)
    mt = lo.dot(up)
    mt = perm_mat(mt, inv_idx(rows), inv_idx(cols))
    print("IsClose=%s" % np.allclose(m0, mt))
    print("NewMat\n%s" % mt)


def test_trans():
    n = 10
    alpha = 10
    bs = [np.arange(n), np.ones(n), np.eye(n)[-1]]
    # mat = mock_mat0(n, alpha)
    mat = np.random.randn(n, n) * alpha + np.eye(n)
    lo = get_lower(mat)
    up = get_upper(mat)
    print("Matrix\n%s" % mat)
    x = np.zeros(n)
    for b in bs:
        copy_arr(b, x)
        ftrans_lower(mat, x)
        b1 = np.dot(lo, x)
        assert np.allclose(b1, b + 1)

    for b in bs:
        copy_arr(b, x)
        btrans_lower(mat, x)
        b1 = np.dot(x, lo)
        assert np.allclose(b1, b + 1)

    for b in bs:
        copy_arr(b, x)
        ftrans_upper(mat, x)
        b1 = np.dot(up, x)
        assert np.allclose(b1, b + 1)

    for b in bs:
        copy_arr(b, x)
        btrans_upper(mat, x)
        b1 = np.dot(x, up)
        assert np.allclose(b1, b + 1)


def test_la():
    n = 100
    alpha = 10
    bs = [np.arange(n, dtype=float), np.ones(n), np.eye(n)[-1]]
    # mat = mock_mat0(n, alpha)
    mat = np.random.randn(n, n) * alpha + np.eye(n)
    m0 = np.copy(mat)
    # print("Matrix\n%s" % mat)
    la = LA()
    la.factor(mat)
    rows = piv2idx(la.p)
    cols = piv2idx(la.q)
    print("P\n%s\n%s\n%s" % (la.p, rows, inv_idx(rows)))
    print("Q\n%s\n%s\n%s" % (la.q, cols, inv_idx(cols)))
    x = np.zeros(n)
    for b in bs:
        copy_arr(b, x)
        la.ftrans(x)
        b1 = np.dot(m0, x)
        assert np.allclose(b1, b)

    for b in bs:
        copy_arr(b, x)
        la.btrans(x)
        b1 = np.dot(x, m0)
        assert np.allclose(b1, b)


if __name__ == '__main__':
    test_la()
    # test_trans()
