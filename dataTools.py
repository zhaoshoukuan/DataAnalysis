import numpy as np
import scipy as sp 
import sympy as sy


xvar = sy.Symbol('x',real=True)


def nearest(y,target,x=None):
    if x is None:
        index = np.argmin(np.abs(y-target))
        return index
    else:
        index = np.argmin(np.abs(y-target))
        return index, x[index]

# def specfunc2data(func,)
def specshift(funclist,fc,bias=0,side='lower'):
    func,voffset, vperiod, ejs, ec, d = funclist
    v = np.linspace(-vperiod/2,vperiod/2,50001) + voffset
    y = sy.lambdify(xvar,func,'numpy')
    if side == 'lower':
        f01 = y(v)[v<voffset]
        index, vtarget = nearest(f01,fc,v[v<voffset])
        # vnew = v - vtarget
        # _, fnew = nearest(vnew,bias,y(v))
        return y(vtarget+bias), (vtarget+bias)
    if side == 'higher':
        f01 = y(v)[v>voffset]
        index, vtarget = nearest(f01,fc,v[v>voffset])
        return y(vtarget+bias), vtarget

def biasshift(funclist,fc,fshift=0,side='lower'):
    func,voffset, vperiod, ejs, ec, d = funclist
    v = np.linspace(-vperiod/2,vperiod/2,50001) + voffset
    y = sy.lambdify(xvar,func,'numpy')
    if np.max(fc+fshift)>np.max(y(v)):
        raise('too big')
    if side == 'lower':
        vnew = v[v<voffset]
        f01 = y(v)[v<voffset]
        finterp = sp.interpolate.interp1d(f01,vnew)
        index, vtarget = nearest(f01,fc,vnew)
        return finterp(fc+fshift)-vtarget
    if side == 'higher':
        vnew = v[v>voffset]
        f01 = y(v)[v>voffset]
        finterp = sp.interpolate.interp1d(f01,vnew)
        return finterp(fc+fshift)