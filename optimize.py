#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimize.py
@Time    :   2020/02/10 16:55:28
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
import time, numpy as np
from scipy import fftpack
from sklearn.cluster import KMeans
from scipy.optimize import least_squares as ls, curve_fit, basinhopping as bh
from scipy import signal
import asyncio

'''
主要采用了全局优化的思想进行拟合
相比于optimize_old模块，该模块的思想是利用全局最优化算法进行数据拟合，方法采自scipy.optimize模块，以basinhopping为例，就是来最小化残差函数，
相比于最小二乘法，他要求残差函数返回的为每一点的残差平方和，该法对初始值相对不敏感，精度高。
'''

################################################################################
### 拟合参数边界
################################################################################

class MyBounds(object):
    def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin
        
################################################################################
### 生数据预处理
################################################################################

class RowToRipe():
    def __init__(self):
        pass
    def space(self,y):
        ymean, ymax, ymin = np.mean(y), np.max(y), np.min(y)
        d = y[np.abs(y-ymean)>0.05*(ymax-ymin)]
        if len(d) < 0.9*len(y):
            return len(d) + int(len(y)*0.1)
        else:
            return len(y)

    def deductPhase(self,x,y):
        if np.ndim(y) != 2:
            y = [y]
        s = []
        for i in y:
            phi = np.unwrap(np.angle(i), 0.9 * np.pi)
            phase = np.poly1d(np.polyfit(x, phi, 1))
            base = i / np.exp(1j * phase(x))
            s.append(base)
        return x, np.array(s)

    def manipulation(self,volt,freq,s):
        s_abs = np.abs(s)        
        min_index = np.argmin(s_abs,axis=1)         
        x, y = np.array(volt), np.array([freq[j] for j in min_index]) 
        return x,y
  
    def firstMax(self,x,y,num=0,peakpercent=0.9,insitu=False):
        index0 = np.argmin(np.abs(x-num))
        y = y - np.min(y)
        peak = peakpercent*y[index0] if insitu else peakpercent*np.max(y)
        c = np.argwhere(y>peak)
        cdiff = np.diff(c[:,0])
        n_clusters = len(np.argwhere(cdiff>np.mean(cdiff))) + 1
        S = c[:,0]
        d = np.mat(list(zip(S,S)))

        kmeans = KMeans(n_clusters=n_clusters,max_iter=100,tol=0.001)
        yfit = kmeans.fit_predict(d)
        index = int(np.mean(S[yfit==yfit[np.argmin(np.abs(S-index0))]]))
        bias0 = round(x[index],3)
        return bias0

    def smooth(self,y,f0=0.1):
        b, a = signal.butter(3,f0)
        z = signal.filtfilt(b,a,y)
        return z

    def resample(self,x,y,num=1001):
        down = len(x)
        up = num
        x_new = np.linspace(min(x),max(x),up)
        z = signal.resample_poly(y,up,down,padtype='line')
        return x_new, z

    def findPeaks(self,y,width=0.02,f0=0.015):
        z = -y
        background = self.smooth(z,f0=f0)
        height = (np.max(z)-np.min(z))
        property_peaks = signal.find_peaks(z,height=(background+0.2*height,background+1.2*height),width=width)
        index, prominences = property_peaks[0], property_peaks[1]['prominences']
        return index, prominences
    
    def spectrum(self,x,y,method='normal',window='boxcar',detrend='constant',axis=-1,scaling='density',average='mean'):
        '''
        scaling:
            'density':power spectral density V**2/Hz
            'spcetrum': power spectrum V**2
        '''
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        if method == 'normal':
            f, Pxx = signal.periodogram(y,fs,window=window,detrend=detrend,axis=axis,scaling=scaling)
        if method == 'welch':
            f, Pxx = signal.welch(y,fs,window=window,detrend=detrend,axis=axis,scaling=scaling,average=average)
        return f, Pxx
    
    def cross_psd(self,x,y,z,window='hann',detrend='constant',scaling='density',axis=-1,average='mean'):
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        f, Pxy = signal.csd(y,z,fs,window=window,detrend=detrend,scaling=scaling,axis=axis,average=average)
        return f, Pxy
    
    def ftspectrum(self,x,y,window='hann',detrend='constant',scaling='density',axis=-1,mode='psd'):
        '''
        mode:
            'psd':
            'complex':==stft
            'magnitude':==abs(stft)
            'angle':with unwrapping
            'phase':without unwraping
        '''
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        f, t, Sxx = signal.spectrigram(y,fs,window=window,detrend=detrend,scaling=scaling,axis=axis,mode=mode)
        return f, t, Sxx
    
    def stft(self,x,y,window='hann',detrend=False,axis=-1,boundary='zeros',padded=True):
        '''
        boundary:you can choose ['even','odd','constant','zeros',None]
        padded: True Or False          
        '''
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        f, t, Zxx = signal.stft(y,fs,window=window,detrend=detrend,axis=axis,boundary=boundary,padded=padded)
        retrun f, t, Zxx
     
    def istft(self,x,Zxx,window='hann',boundary=True,time_axis=-1,freq_axis=-2):
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        t, y = signal.stft(Zxx,fs,window=window,boundary=boundary,time_axis=time_axis,freq_axis=freq_axis)
        retrun t, y

    def fourier(self,x,y):
        sample = (np.max(x) - np.min(x))/(len(x) - 1)
        yt  = np.fft.fftshift(np.fft.fftfreq(len(y))) / sample
        amp = np.fft.fftshift(np.fft.fft(y))
        w = np.abs(yt[yt!=0][np.argmax(np.abs(amp[yt!=0]))])
        return w, yt[yt!=0], np.abs(amp[yt!=0])
        
    def envelope(self,y):
        mold, out, rc = 0, [], self.responsetime
        out.append(np.abs(y[0]))
        for j, i in enumerate(y[1:],start=1):
            i = np.abs(i)
            if i > out[j-1]:
                mold = i
            else:
                mold = (out[j-1] * rc)/(rc + 1)
            out.append(mold)
        return out

    def envelope_Hilbert(self,y):
        ym = y - y.mean()
        yh = fftpack.hilbert(ym) 
        out = np.abs(ym + 1j*yh) + y.mean()
        return out

    def profile(self,v,f,s,peak,axis=1,classify=False):
        if classify:
            index = np.argwhere(np.abs(s)>peak)
            v = v[index[:,0]]
            f = f[index[:,1]]
        else:
            if axis == 1:
                v = v[np.abs(s).max(axis=1)>peak]
                s = s[np.abs(s).max(axis=1)>peak]
                f = f[np.abs(s).argmax(axis=1)]
            if axis == 0:
                f = f[np.abs(s).max(axis=0)>peak]
                s = s[:,np.abs(s).max(axis=0)>peak]
                v = v[np.abs(s).argmax(axis=0)]
        return v, f

################################################################################
### 拟合Exp函数
################################################################################

class Exp_Fit(RowToRipe):
    
    def __init__(self,funcname=None):
        self.funcname = funcname
    
    def errExp(self,paras, x, y):
        
        if self.funcname == 'gauss':
            A, B, T1, T2 = paras
            return np.sum((A * np.exp(-T2*x**2-x*T1) + B - y)**2)
        else:
            A, B, T1 = paras
            return np.sum((A * np.exp(-x*T1) + B - y)**2)

    def guessExp(self,x,y):
        ymin = y.min()
        y = y-y.min()
        mask = y > 0.05*y.max()
        if self.funcname == 'gauss':
            a = np.polyfit(x[mask], np.log(y[mask]), 2)
            return [np.exp(a[2]), ymin, -a[1], -a[0]]
        else:
            a = np.polyfit(x[mask], np.log(y[mask]), 1)
            return [np.exp(a[1]), ymin, -a[0]]

    def fitExp(self,x,y):
        p0 = self.guessExp(x,y)
        # res = ls(self.errExp, p0, args=(x, y)) 
        res = bh(self.errExp,p0,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)}) 
        return res.x

################################################################################
### 拟合Cos函数
################################################################################

class Cos_Fit(RowToRipe):

    def __init__(self):
        pass

    def errCos(self,paras,x,y):
        A,C,W,phi = paras             
        return  np.sum((A*np.cos(2*np.pi*W*x+phi)+C-y)**2)  

    def guessCos(self,x,y):
        x, y = np.array(x), np.array(y)
        sample = (np.max(x) - np.min(x))/(len(x) - 1)
        Ag, Cg= np.max(y)-np.min(y), np.mean(y) 
        yt  = np.fft.fftshift(np.fft.fftfreq(len(y))) / sample
        amp = np.fft.fftshift(np.fft.fft(y))
        z = np.abs(amp[yt!=0])
        ytz = yt[yt!=0]
        Wg = np.abs(ytz[np.argmax(z)])
        phig =  np.mean(np.arccos((y[0] - Cg)/Ag) - 2*np.pi*Wg*x[0])
        return Ag, Cg, Wg, phig

    def fitCos(self,volt,s):
        x, y = volt, s
        if x[0] / 1e9 > 1:
            raise 'I hate the large number, please divided by 1e9, processing x in GHz'
        Ag, Cg, Wg, phig = self.guessCos(x,y)
        p0 = Ag, Cg, Wg, phig
        # print(Ag, Cg, Wg, phig)
        # res = ls(self.errCos, [Ag,Cg,Wg,phig], args=(x, y)) 
        mybounds = MyBounds(xmin=[-np.inf,-np.inf,0,-np.pi],xmax=[np.inf,np.inf,1.5*Wg,np.pi])    
        res = bh(self.errCos,p0,niter=80,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)},accept_test=mybounds)          
        A, C, W, phi = res.x
        return A, C, W, phi

################################################################################
### 拟合洛伦兹函数
################################################################################

class Lorentz_Fit(RowToRipe):
    '''
    I hate the large number
    processing x in GHz
    '''
    def __init__(self):
        pass

    def errLorentz(self,para,x,y):
        a,b,c,d = para
        return np.sum((a/(1.0+c*(x-b)**2)+d-y)**2)

    def guessLorentz(self,x,y):
        z = np.sort(np.abs(y))
        d = np.mean(z[:int(len(z)/2)])
        y = np.abs(y)- d
        b = x[np.abs(y).argmax()]
        bw = (np.max(x[y>0.5*(np.max(y)-np.min(y))])-np.min(x[y>0.5*(np.max(y)-np.min(y))]))/2
        a = np.abs(y).max()
        c = 1 / bw**2
        return a,b,c,d

    def fitLorentz(self,x,y):
        if x[0] / 1e9 > 1:
            raise 'I hate the large number, please divided by 1e9, processing x in GHz'
        para = self.guessLorentz(x,y)
        # mybounds = MyBounds(xmin=[-np.inf,-np.inf,-np.inf,-np.inf,0,0],xmax=[np.inf,np.inf,np.inf,np.inf,1.5*w,2*np.pi])    
        res = bh(self.errLorentz,para,niter=50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)})
        # res = ls(self.errLorentz,para,args=(x,y))
        a,b,c,d = res.x
        return a,b,c,d,np.sqrt(np.abs(1/c))*2e3

################################################################################
### 拟合指数包络函数
################################################################################

class T2_Fit(Exp_Fit,Cos_Fit):
    '''
    #############
    example:
    import imp
    import optimize
    op = imp.reload(optimize)
    try: 
        fT2 = op.T2_Fit(funcname='gauss',envelopemethod='hilbert')
        A,B,T1,T2,w,phi = fT2.fitT2(t,y)
    finally:
        pass
    ##############
    '''
    def __init__(self,responsetime=100,T1=35000,phi=0,funcname=None,envelopemethod=None):
        Exp_Fit.__init__(self,funcname)
        self.responsetime = responsetime
        self.T1 = T1
        self.phi = phi
        self.envelopemethod = envelopemethod
    
    def guessT2(self,x,y_new,y):
 
        A, B, T1, T2 = self.fitExp(x[5:-5],y_new[5:-5])
        T1 = 1 / T1 / 2
        if np.abs(self.T1-T1)>5000:
            T1 = self.T1
        Ag, Cg, Wg, phig = self.guessCos(x,y)
        return A, B, T1, np.sqrt(np.abs(1/T2)), Wg, phig

    def errT2(self,para,x,y):
        A,B,T1,T2,w,phi = para
        return np.sum((A*np.exp(-(x/T2)**2-x/T1/2)*np.cos(2*np.pi*w*x+phi) + B - y)**2)

    def fitT2(self,x,y):
        '''
        几个参数的限制范围还需要考究，A，T1，T2
        '''
        d = self.space(y)
        if self.envelopemethod == 'hilbert':
            out = self.envelope_Hilbert(y)
        else:
            out = self.envelope(y)
        A,B,T1,T2,w,phi = self.guessT2(x,out,y)
        env = A,B,T1,T2,out
        if T2 > 0.8*x[d-1] and d < 0.8*len(y):
            T2 = 0.37*x[d-1]
        amp = (np.max(y)-np.min(y)) / 2
        A = A if np.abs(A-amp) < 0.1*amp else amp
        p0 = A,B,T1,T2,w,self.phi
        print(p0)
        # res = ls(self.errT2, p0, args=(x, y)) 
        mybounds = MyBounds(xmin=[-np.inf,-np.inf,0,0,0,-np.pi],xmax=[np.inf,np.inf,100000,10000,1.5*w,np.pi])    
        res = bh(self.errT2,p0,niter = 80,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)},accept_test=mybounds)     
        A,B,T1,T2,w,phi = res.x
        return A,B,T1,T2,w,phi,env

class Rabi_Fit(T2_Fit):

    def __init__(self,responsetime=100,T1=20000,phi=np.pi/2,funcname=None,envelopemethod=None):
        T2_Fit.__init__(self,responsetime,T1,phi,funcname,envelopemethod)
        
    
    def guessRabi(self,x,y_new,y):
 
        A, B, T1 = self.fitExp(x[5:-5],y_new[5:-5])
        T1 = 1 / T1
        if np.abs(self.T1-T1)>5000:
            T1 = self.T1
        Ag, Cg, Wg, phig = self.guessCos(x,y)
        return A, B, T1, Wg, phig

    def errRabi(self,para,x,y):
        A,B,T1,w,phi = para
        return np.sum((A*np.exp(-x/T1)*np.cos(2*np.pi*w*x+phi) + B - y)**2)

    def fitRabi(self,x,y):
        if self.envelopemethod == 'hilbert':
            out = self.envelope_Hilbert(y)
        else:
            out = self.envelope(y)
        A,B,T1,w,phi = self.guessRabi(x,out,y)
        env = (A,B,T1,out)
        amp = (np.max(y)-np.min(y)) / 2
        A = A if np.abs(A-amp) < 0.1*amp else amp
        B = B if np.abs(B-np.mean(y)) < 0.1*np.mean(y) else np.mean(y)
        p0 = A,B,T1,w,self.phi
        print(p0)
        # res = ls(self.errRabi, p0, args=(np.array(x), np.array(y)))   
        mybounds = MyBounds(xmin=[-np.inf,-np.inf,0,0,0],xmax=[np.inf,np.inf,100e3,1.5*w,2*np.pi])
        res = bh(self.errRabi,p0,niter=30,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)},accept_test=mybounds)      
        A,B,T1,w,phi = res.x
        return A,B,T1,w,phi,env

################################################################################
### 拟合二维谱
################################################################################

class Spec2d_Fit(Cos_Fit):

    def __init__(self,peak=15):
        self.peak = peak
    
    def profile(self,v,f,s,classify=False):
        if classify:
            index = np.argwhere(np.abs(s)>self.peak)
            v = v[index[:,0]]
            f = f[index[:,1]]
        else:
            v = v[np.abs(s).max(axis=1)>self.peak]
            s = s[np.abs(s).max(axis=1)>self.peak]
            f = f[np.abs(s).argmax(axis=1)]
        return v, f
    def err(self,paras,x,y):
        A, C, w, phi = paras
        return np.sum((np.sqrt(A*np.abs(np.cos(w*x+phi))) + C - y)**2)
    def fitSpec2d(self,v,f,s,classify=False):
        v,f = self.profile(v,f,s,classify)
        A, C, W, phi = self.fitCos(v,f)
        mybounds = MyBounds(xmin=[0,-np.inf,0,0],xmax=[15*np.abs(A),np.inf,2.5*W,2*np.pi])
        res = bh(self.err,[np.abs(A),C,2*W,phi],niter = 100,minimizer_kwargs={"method":"Nelder-Mead","args":(v, f)},accept_test=mybounds) 
        A, C, W, phi = res.x
        return f,v,A, C, W, phi

################################################################################
### crosstalk直线拟合
################################################################################

class Crosstalk_Fit(Spec2d_Fit):

    def __init__(self,peak=15):
        self.peak = peak

    def fitCrosstalk(self,v,f,s,classify=False):
        v,f = self.profile(v,f,s,classify)
        res = np.polyfit(f,v,1)
        return v, f, res

################################################################################
### 单比特tomo
################################################################################

def pTorho(plist):
    pz_list, px_list, py_list = plist
    rho_list = []
    for i in range(np.shape(pz_list)[0]):
        pop_z, pop_x, pop_y = pz_list.T[i], px_list.T[i], py_list.T[i]
        rho_00, rho_01 = 1 - pop_z, (2*pop_x - 2j*pop_y - 1 + 1j) / 2j
        rho_10, rho_11 = (1 + 1j - 2*pop_x - 2j*pop_y) / 2j, pop_z
        rho = np.array([[rho_00,rho_01],[rho_10,rho_11]])
        rho_list.append(rho)
    pass

################################################################################
### RB
################################################################################

class RB_Fit:
    def __init__(self):
        pass
    def err(self,paras,x,y):
        A,B,p = paras
        return A*p**x+B-y
    def guess(self,x,y):
        B = np.min(y)
        y = y - np.min(y)
        mask = y > 0
        a = np.polyfit(x[mask], np.log(y[mask]), 1)
        return np.exp(np.abs(a[1])), B, 1/np.exp(np.abs(a[0]))
    def fitRB(self,x,y):
        p0 = self.guess(x,y)
        res = ls(self.err, p0, args=(x, y)) 
        A,B,p = res.x
        return A, B, p

################################################################################
### 双指数拟合
################################################################################

class TwoExp_Fit(Exp_Fit):
    def __init__(self,funcname=None,percent=0.2):
        Exp_Fit.__init__(self,funcname)
        self.percent = percent
    def err(self,paras,x,y):
        a, b, c, d, e = paras
        return np.sum((a*np.exp(b*x) + c*np.exp(d*x) + e - y)**2)
    def guess(self,x,y):
        a,e,b = self.fitExp(x,y)
        b *= -1
        e = np.min(y) if a > 0 else np.max(y)
        return a,b,a*self.percent,b*self.percent,e
    def fitTwoexp(self,x,y):
        p0 = self.guess(x,y)
        a, b, c, d, e = p0
        lower = [0.95*i if i > 0 else 1.05*i for i in p0]
        higher = [1.05*i if i > 0 else 0.95*i for i in p0]
        lower[2], lower[3] = -np.abs(a)*self.percent, -np.abs(b)*self.percent
        higher[2], higher[3] = self.percent*np.abs(a), self.percent*np.abs(b)
        print(p0)
        # res = ls(self.err,p0,args=(x,y),bounds=(lower,higher))
        res = bh(self.err,p0,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)})

        return res.x

################################################################################
## 真空拉比拟合
################################################################################

class Vcrabi_fit():
    def __init__(self):
        pass
    def err(self,paras,x,y):
        g, A0, Z0 = paras
        return np.sum((np.sqrt(4*(g/2/np.pi)**2+A0**2*(x-Z0)**2)-y)**2)
    def guess(self,x,y):
        Z0 = x[np.argmin(y)]
        g = np.min(y)*np.pi
        x, y = x[x!=Z0], y[x!=Z0]
        A0 = np.mean(np.sqrt(y**2-4*(g/2/np.pi)**2)/(x-Z0))
        return g, A0, Z0
    def fitVcrabi(self,x,y):
        p0 = self.guess(x,y)
        print(p0)
        res = bh(self.err,p0,niter=50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)})
        return res.x

################################################################################
## 拟合Q值
################################################################################

class Cavity_fit(RowToRipe):
    def __init__(self):
        pass

    def circleLeastFit(self,x, y):
        def circle_err(params, x, y):
            xc, yc, R = params
            return (x - xc)**2 + (y - yc)**2 - R**2

        p0 = [
            x.mean(),
            y.mean(),
            np.sqrt(((x - x.mean())**2 + (y - y.mean())**2).mean())
        ]
        res = ls(circle_err, p0, args=(x, y))
        return res.x

    def guessParams(self,x,s):
        
        y = np.abs(1 / s)
        f0 = x[y.argmax()]
        _bw = x[y > 0.5 * (y.max() + y.min())]
        FWHM = np.max(_bw) - np.min(_bw)
        Qi = f0 / FWHM
        _, _, R = self.circleLeastFit(np.real(1 / s), np.imag(1 / s))
        Qe = Qi / (2 * R)
        QL = 1 / (1 / Qi + 1 / Qe)

        return [f0, Qi, Qe, 0, QL]

    def invS21(self, f, f0, Qi, Qe, phi):
        #QL = 1/(1/Qi+1/Qe)
        return 1 + (Qi / Qe * np.exp(1j * phi)) / (
            1 + 2j * Qi * (np.abs(f) / np.abs(f0) - 1))
    
    def err(self,params,f,s21):
        f0, Qi, Qe, phi = params
        y = np.abs(s21) - np.abs(self.invS21(f, f0, Qi, Qe, phi) )
        return np.sum(np.abs(y)**2)

    def fitCavity(self,x,y):
        f, s = self.deductPhase(x,y)
        s = s[0]/np.max(np.abs(s[0]))
        f0, Qi, Qe, phi, QL = self.guessParams(f,s)
        res = bh(self.err,(f0, Qi, Qe, phi),niter = 100,\
            minimizer_kwargs={"method":"Nelder-Mead","args":(f, 1/s)}) 
        f0, Qi, Qe, phi = res.x
        QL = 1 / (1 / Qi + 1 / Qe)
        return f0, Qi, Qe, QL, phi, f, s
