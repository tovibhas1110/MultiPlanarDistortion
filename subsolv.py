import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import pdb

def subsolv(m, n, epsimin, low, upp, alfa, beta, 
            p0, q0, P, Q, a0, a, b, c, d):
    
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.ones(1)
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.ones(1)
    s = eem.copy()
    itera = 0

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        
        ux1 = upp - x
        xl1 = x - low
        
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        
        gvec = P @ uxinv1 + Q @ xlinv1
        dpsidx = plam / ux2 - qlam / xl2
        
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        

        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm



        
        residu1 = np.concatenate([rex.T, rey.T, rez.reshape(-1, 1)], axis=1)
        residu1 = np.reshape(residu1,(residu1.size,1), order ='F')
        residu2 = np.concatenate([relam.T.reshape(-1), rexsi.T.reshape(-1), reeta.T.reshape(-1), remu.T.reshape(-1), rezet.T.reshape(-1), res.T.reshape(-1)])
        residu2 = np.reshape(residu2,(residu2.size,1), order ='F')
        residu = np.concatenate([residu1.T, residu2.T], axis=1)
        residu = np.reshape(residu,(residu.size,1), order ='F')
        residunorm = np.sqrt(residu.T @ residu)
        residumax = np.max(np.abs(residu))
        
        ittt = 0

        while residumax > 0.9 * epsi and ittt < 100:
            ittt += 1
            itera += 1
            
            if ittt == 100:
                print('max inner iter reached')
            
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1
            GG = P @ diags(uxinv2.T.reshape(-1),0) - Q @ diags(xlinv2.T.reshape(-1),0);
            dpsidx = plam/ux2 - qlam/xl2
            delx = dpsidx - epsvecn/(x-alfa) + epsvecn/(beta-x)
            dely = c + d*y - lam - epsvecm/y
            delz = a0 - a.T @ lam - epsi/z
            dellam = gvec - (a * z) - y - b + epsvecm/lam
            diagx = plam/ux3 + qlam/xl3
            diagx = 2*diagx + xsi/(x-alfa) + eta/(beta-x)
            diagxinv = een/diagx
            diagy = d + mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely/diagy - GG@(delx/diagx)
                bb = np.concatenate([blam.T.reshape(-1),delz.T.reshape(-1)])
                bb = np.reshape(bb,(bb.size,1), order ='F')
                Alam = diags(diaglamyi.T.reshape(-1),0).toarray() + GG @ diags(diagxinv.T.reshape(-1),0).toarray() @ GG.T
                AA = np.concatenate([np.concatenate([Alam, a],axis=1),np.concatenate([a.T.reshape(-1), (-zet/z).T.reshape(-1)]).reshape(1,-1)],axis=0)
                solut = np.linalg.solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m]
                dx = -delx/diagx - (GG.T @ dlam)/diagx
                
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam + dely/diagy
                Axx = diags(diagx.T.reshape(-1),0).toarray() + GG.T @ spdiags(diaglamyiinv.T.reshape(-1),0).toarray() @ GG
                azz = zet/z + a.T @ (a/diaglamyi)
                axz = -GG.T @ (a/diaglamyi)
                bx = delx + GG.T @ (dellamyi/diaglamyi)
                bz  = delz - a.T @ (dellamyi/diaglamyi)
                AA = np.concatenate([np.concatenate([Axx, axz],axis=1),np.concatenate([axz.T.reshape(-1), [azz]]).reshape(1,-1)],axis=0)
                bb = np.concatenate([-bx.T.reshape(-1),-bz.T.reshape(-1)])
                bb = np.reshape(bb,(bb.size,1), order ='F')
                solut = np.linalg.solve(AA, bb)
                dx  = solut[0:n]
                dz = solut[n].item()
                dlam = (GG @ dx)/diaglamyi - dz @ (a/diaglamyi) + dellamyi/diaglamyi

            dy = -dely/diagy + dlam/diagy
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
            dmu  = -mu + epsvecm/y - (mu*dy)/y
            dzet = -zet + epsi/z - zet * (dz/z)
            ds   = -s + epsvecm/lam - (s*dlam)/lam
            xx  = np.concatenate([ y.T.reshape(-1),  z.T.reshape(-1),  lam.T.reshape(-1),  xsi.T.reshape(-1),  eta.T.reshape(-1),  mu.T.reshape(-1),  zet.T.reshape(-1) , s.T.reshape(-1)]).reshape(-1,1)
            dxx = np.concatenate([dy.T.reshape(-1), dz.T.reshape(-1), dlam.T.reshape(-1), dxsi.T.reshape(-1), deta.T.reshape(-1), dmu.T.reshape(-1), dzet.T.reshape(-1), ds.T.reshape(-1)]).reshape(-1,1)

            stepxx = -1.01*dxx/xx
            stmxx  = np.max(stepxx,axis=0)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = np.max(stepalfa,axis=0)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = np.max(stepbeta,axis=0)
            stmalbe  = np.maximum(stmalfa,stmbeta)
            stmalbexx = np.maximum(stmalbe,stmxx)
            stminv = np.maximum(stmalbexx,1)
            steg = (1/stminv.item())

            xold   =   x
            yold   =   y
            zold   =   z
            lamold =  lam
            xsiold =  xsi
            etaold =  eta
            muold  =  mu
            zetold =  zet
            sold   =   s

            itto = 0
            resinew = 2*residunorm

            while resinew > residunorm and itto < 50:
                itto = itto+1
                
                x   =   xold + steg * dx
                y   =   yold + steg * dy
                z   =   zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu  = muold  + steg * dmu
                zet = zetold + steg * dzet
                s   =   sold + steg * ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam/ux2 - qlam/xl2

                rex = dpsidx - xsi + eta
                rey = c + d*y - mu - lam
                rez = a0 - zet - a.T @ lam
                relam = gvec - a @ z - y + s - b
                rexsi = xsi*(x-alfa) - epsvecn
                reeta = eta*(beta-x) - epsvecn
                remu = mu*y - epsvecm
                rezet = zet@z - epsi
                res = lam*s - epsvecm

                
                residu1 = np.concatenate([rex.T, rey.T, rez.reshape(-1, 1)], axis=1)
                residu1 = np.reshape(residu1,(residu1.size,1), order ='F')
                residu2 = np.concatenate([relam.T.reshape(-1), rexsi.T.reshape(-1), reeta.T.reshape(-1), remu.T.reshape(-1), [rezet], res.T.reshape(-1)])
                residu2 = np.reshape(residu2,(residu2.size,1), order ='F')
                residu = np.concatenate([residu1.T, residu2.T], axis=1)
                residu = np.reshape(residu,(residu.size,1), order ='F')
                resinew = np.sqrt(residu.T @ residu)
                steg = steg/2

            residunorm = resinew
            residumax = np.max(np.abs(residu))
            steg = 2*steg

        epsi = 0.1*epsi
    
    xmma   =   x
    ymma   =   y
    zmma   =   z
    lamma =  lam
    xsimma =  xsi
    etamma =  eta
    mumma  =  mu
    zetmma =  zet
    smma   =   s

    return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma