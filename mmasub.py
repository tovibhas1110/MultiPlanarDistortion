import numpy as np
from scipy.sparse import diags
from subsolv import subsolv
import pdb

def mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, df0dx2,
           fval, dfdx, dfdx2,
           low, upp,
           a0mma, amma, cmma, dmma):
        
        epsimin = np.sqrt(m + n) * 1e-9
        feps = 1e-6
        asyinit = 0.5
        asyincr = 1.2
        asydecr = 0.7
        albefa = 0.1
        een = np.reshape(np.ones(n),(n,1),order = 'F')       # Column vector equivalent is just a 1D array in NumPy
        zeron = np.reshape(np.zeros(n),(n,1),order = 'F')
        iter = loop

        if iter < 2.5:
             low = xval - asyinit * (xmax - xmin)
             upp = xval + asyinit * (xmax - xmin)
        else:
             zzz = (xval - xold1) * (xold1 - xold2)  # element-wise multiplication
             #print("zzz:",zzz)
             #print("xval:",xval)
             #print("xold1:",xold1)
             #print("xold2:",xold2)
             #breakpoint()
             factor = np.full_like(xval, een)  # start with 'een' as default
             factor[zzz > 0] = asyincr
             factor[zzz < 0] = asydecr
             low = xval - factor * (xold1 - low)
             upp = xval + factor * (upp - xold1)
             lowmin = xval - 10 * (xmax - xmin)
             lowmax = xval - 0.01 * (xmax - xmin)
             uppmin = xval + 0.01 * (xmax - xmin)
             uppmax = xval + 10 * (xmax - xmin)
             low = np.maximum(low, lowmin)
             low = np.minimum(low, lowmax)
             upp = np.minimum(upp, uppmax)
             upp = np.maximum(upp, uppmin)
        
        zzz = low + albefa * (xval - low)
        alfa = np.maximum(zzz, xmin)
        zzz = upp - albefa * (upp - xval)
        beta = np.minimum(zzz, xmax)
        
        ux1 = upp - xval
        ux2 = ux1 * ux1
        ux3 = ux2 * ux1
        xl1 = xval - low
        xl2 = xl1 * xl1
        xl3 = xl2 * xl1
        ul1 = upp - low
        ulinv1 = een / ul1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        uxinv3 = een / ux3
        xlinv3 = een / xl3
        diap = (ux3 * xl1) / (2 * ul1)
        diaq = (ux1 * xl3) / (2 * ul1)

        p0 = zeron.copy()
        p0[df0dx > 0] = df0dx[df0dx > 0]
        p0 = p0 + 0.001 * np.abs(df0dx) + feps * ulinv1
        p0 = p0 * ux2  # element-wise multiplication
        
        q0 = zeron.copy()             # Copy to avoid modifying the original array
        q0[df0dx < 0] = -df0dx[df0dx < 0]   # Boolean indexing replaces find()
        q0 = q0 + 0.001 * np.abs(df0dx) + feps * ulinv1
        q0 = q0 * xl2                 # Element-wise multiplication
        
        dg0dx2 = 2 * (p0 / ux3 + q0 / xl3)
        del0 = df0dx2 - dg0dx2
        delpos0 = np.zeros_like(del0)
        delpos0[del0 > 0] = del0[del0 > 0]
        
        p0 = p0 + delpos0 * diap
        q0 = q0 + delpos0 * diaq
        


        # Initialize P and Q
        P = np.zeros((m, n))
        P[dfdx > 0] = dfdx[dfdx > 0]
        P = P @ diags(ux2.T.reshape(-1), 0).toarray()   # Equivalent to MATLAB P * spdiags(ux2,0,n,n)
        Q = np.zeros((m, n))
        Q[dfdx < 0] = -dfdx[dfdx < 0]
        Q = Q @ diags(xl2.T.reshape(-1), 0).toarray()   # Equivalent to MATLAB Q * spdiags(xl2,0,n,n)

        dgdx2 = 2*(P @ diags(uxinv3.T.reshape(-1), 0).toarray() + Q @ diags(xlinv3.T.reshape(-1), 0).toarray())
        del_ = dfdx2 - dgdx2   # use del_ since 'del' is a Python keyword
        delpos = np.zeros((m, n))
        delpos[del_ > 0] = del_[del_ > 0]

        # Update P and Q
        P = P + delpos @ diags(diap.T.reshape(-1), 0).toarray()
        Q = Q + delpos @ diags(diaq.T.reshape(-1), 0).toarray()

        # Final computation
        b = P @ uxinv1 + Q @ xlinv1 - fval

        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
            m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0mma, amma, b, cmma, dmma)
        
        return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp