from pymatting import cg, load_image, trimap_split, cf_laplacian, ichol, jacobi, vcycle, CounterCallback
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import scipy.io

# load images resized to 20% of their original size
size = 0.1
image = load_image("data/lemur.png", "RGB", size, "BILINEAR")
trimap = load_image("data/lemur_trimap.png", "GRAY", size, "NEAREST")
h, w = image.shape[:2]
n = h * w

# make closed form laplacian
L = cf_laplacian(image, epsilon=1e-7)

# assembly linear system A x = b
lambd = 100.0
is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)
A = L + scipy.sparse.diags(lambd * is_known.flatten())
b = lambd * is_fg.flatten()

# solve with various preconditioners
for precondition, name in [
    (ichol(A), "ichol"),
    (vcycle(A, (h, w)), "vcycle"),
    (jacobi(A), "jacobi"),
    (None, "none"),
]:
    callback = CounterCallback()
    x = cg(A, b, maxiter=10000, rtol=1e-6, atol=0, M=precondition, callback=callback)
    print(f"{name:6s}: {callback.n} iterations")

x_true = scipy.sparse.linalg.spsolve(A, b)

print("")
print("sum(abs(x - A^-1 b)) =", np.sum(np.abs(x - x_true)))

# transpose and reshape to matlab format
inds = np.arange(n).reshape(h, w).T.flatten()
A = A[inds, :][:, inds]
x_true = x_true[inds].reshape(n, 1)
b = b[inds].reshape(n, 1)

# save
scipy.io.savemat("data/alphamatting.mat", mdict={
    'A': A,
    'b': b,
    'x_true': x_true,
    'rows': h,
    'cols': w,
})

# display result
alpha = np.clip(x, 0, 1).reshape(h, w)

plt.imshow(alpha, cmap='gray')
plt.show()
