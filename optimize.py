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
import time, numpy as np, matplotlib.pyplot as plt
from scipy import fftpack
from sklearn.cluster import KMeans
from scipy.optimize import least_squares as ls, curve_fit, basinhopping as bh
from scipy import signal
import asyncio, scipy
import qulab.dataTools as dt

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
    
    def errspace(self,func,paras,args):

        return (func(args['x'],paras)-args['y'])**2

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
        bias0 = round(x[index],5)
        return bias0

    def smooth(self,y,f0=0.1,axis=-1):
        b, a = signal.butter(3,f0)
        z = signal.filtfilt(b,a,y,axis=axis)
        return z

    def resample(self,x,y,num=1001):
        down = len(x)
        up = num
        x_new = np.linspace(min(x),max(x),up)
        z = signal.resample_poly(y,up,down,padtype='line')
        return x_new, z

    def findPeaks(self,y,width=None,f0=0.015,h=0.15,threshold=None,prominence=None,plateau_size=None,rel_height=0):
        detrend = np.mean(y - signal.detrend(y))
        # mask = y > (np.max(y)+np.min(y))/2
        z = y if np.max(y)-detrend>detrend-np.min(y) else -y
        background = self.smooth(z,f0=f0)
        height0 = (np.max(z)-np.min(z))
        height = (background+h*height0,background+(1+h)*height0)
        threshold = threshold if threshold == None else threshold*height0
        property_peaks = signal.find_peaks(z,height=height,threshold=threshold,plateau_size=plateau_size)
        index = property_peaks[0]
        half_widths = signal.peak_widths(z,index,rel_height=rel_height)
        print(index,half_widths[0])
        side = (index+int(half_widths[0]), index-int(half_widths[0]))
        prominence = signal.peak_prominences(z,index)
        return index, side, prominence
    
    def spectrum(self,x,y,method='normal',window='boxcar',detrend='constant',axis=-1,scaling='density',average='mean',shift=True):
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
        f, Pxx = (np.fft.fftshift(f), np.fft.fftshift(Pxx)) if shift else (f, Pxx)
        index = np.argmax(Pxx,axis=axis)
        w = f[index]
        return w, f, Pxx
    
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
        return f, t, Zxx
     
    def istft(self,x,Zxx,window='hann',boundary=True,time_axis=-1,freq_axis=-2):
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        t, y = signal.stft(Zxx,fs,window=window,boundary=boundary,time_axis=time_axis,freq_axis=freq_axis)
        return t, y


    def fourier(self,x,y,axis=-1,shift=True):
        y = signal.detrend(y,axis=axis)
        sample = (np.max(x) - np.min(x))/(len(x) - 1)
        # sample = 100
        if shift:
            yt  = np.fft.fftshift(np.fft.fftfreq(np.shape(y)[axis])) / sample
            amp = np.fft.fftshift(np.fft.fft(y,axis=axis))
        else:
            yt  = np.fft.fftfreq(np.shape(y)[axis]) / sample
            amp = np.fft.fft(y,axis=axis)
        w = np.abs(yt[np.argmax(np.abs(amp),axis=axis)])
        # w = self.firstMax(yt,np.abs(amp),peakpercent=0.8)
        return w, yt, np.abs(amp)
        
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

    def poly(self,x,y,num=1):
        z = np.polyfit(x, y, num)
        func = np.poly1d(z)
        return z, func

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

    def func(self,x,paras):
        A,C,W,phi = paras  
        return A*np.cos(2*np.pi*W*x+phi)+C

    def errCos(self,paras,x,y):             
        return  np.sum((self.func(x,paras)-y)**2)  

    def guessCos(self,x,y):
        x, y = np.array(x), np.array(y)
        # sample = (np.max(x) - np.min(x))/(len(x) - 1)
        Ag, Cg= np.abs(y-np.mean(y)).max(), np.mean(y) 
        # yt  = np.fft.fftshift(np.fft.fftfreq(len(y))) / sample
        # amp = np.fft.fftshift(np.fft.fft(y))
        Wg,yt,amp = RowToRipe().fourier(x, y)
        z = np.abs(amp[yt!=0])
        ytz = yt[yt!=0]
        # Wg = np.abs(ytz[np.argmax(z)])
        phig =  np.mean(np.arccos((y - Cg)/Ag) - 2*np.pi*Wg*x) % (2*np.pi)
        return Ag, Cg, Wg, 0

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
        return res, self.func

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
        # index, prominences, widths = self.findPeaks(y)
        z = np.sort(np.abs(y))
        d = np.mean(z[:int(len(z)/2)])
        y = np.abs(y)- d
        b = x[np.abs(y).argmax()]
        # b1, b = x[index]
        bw = (np.max(x[y>0.5*(np.max(y)-np.min(y))])-np.min(x[y>0.5*(np.max(y)-np.min(y))]))/2
        # bw1, bw = widths
        a = np.abs(y).max()
        # a1, a = prominences
        c = 1 / bw**2
        return a,b,c,d

    def fitLorentz(self,x,y):
        if x[0] / 1e9 > 1:
            raise 'I hate the large number, please divided by 1e9, processing x in GHz'
        para = self.guessLorentz(x,y)
        # mybounds = MyBounds(xmin=[-np.inf,-np.inf,-np.inf,-np.inf,0,0],xmax=[np.inf,np.inf,np.inf,np.inf,1.5*w,2*np.pi])    
        res = bh(self.errLorentz,para,niter=20,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)})
        # res = ls(self.errLorentz,para,args=(x,y))
        a,b,c,d = res.x
        return res,np.sqrt(np.abs(1/c))*2e3

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

    def func(self,x,para):
        A,B,T1,T2,w,phi = para
        return A*np.exp(-(x/T2)**2-x/T1/2)*np.cos(2*np.pi*w*x+phi) + B

    def errT2(self,para,x,y):
        return np.sum((self.func(x,para) - y)**2)

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
        mybounds = MyBounds(xmin=[0,-np.inf,100,10,0,-np.pi],xmax=[np.inf,np.inf,100000,100000,1.5*w,np.pi])    
        res = bh(self.errT2,p0,niter = 80,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)},accept_test=mybounds)     
        A,B,T1,T2,w,phi = res.x
        return res, self.func

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
        mybounds = MyBounds(xmin=[0,-np.inf,100,0,0],xmax=[np.inf,np.inf,100e3,1.5*w,2*np.pi])
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
            index = np.argwhere(s>self.peak)
            v = v[index[:,0]]
            f = f[index[:,1]]
        else:
            v = v[s.max(axis=1)>self.peak]
            s = s[s.max(axis=1)>self.peak]
            f = f[s.argmax(axis=1)]
        return v, f
    def f01(self,x,paras):
        voffset, vperiod, ejs, ec, d = paras
        tmp = np.pi*(x-voffset)/vperiod
        f01 = np.sqrt(8*ejs*ec*np.abs(np.cos(tmp))*np.sqrt(1+d**2*np.tan(tmp)**2))-ec
        return f01
    def err(self,paras,x,y):
        # A, C, w, phi = paras
        # f01 = np.sqrt(A*np.abs(np.cos(w*x+phi))) + C
        voffset, vperiod, ejs, ec, d = paras
        tmp = np.pi*(x-voffset)/vperiod
        f01 = np.sqrt(8*ejs*ec*np.abs(np.cos(tmp))*np.sqrt(1+d**2*np.tan(tmp)**2))-ec
        return np.sum((f01 - y)**2)

    def fitSpec2d(self,v,f,s=None,classify=False):
        if s is not None:
            v,f = self.profile(v,f,s,classify)
        paras, func = self.fitCos(v,f)
        A, C, W, phi = paras.x
        voffset, vperiod, ec, d = self.firstMax(v,f,num=0), 1/W, 0.2, 0
        ejs = (np.max(f)+ec)**2/8/ec
        p0 = [voffset, vperiod,ejs,ec,d]

        while 1:
            # print(p0)
            mybounds = MyBounds(xmin=[0.5*voffset,0,0,0,0],xmax=[1.5*voffset,1.5*vperiod,2*ejs,2*ec,10])
            res = bh(self.err,p0,niter = 200,minimizer_kwargs={"method":"Nelder-Mead","args":(v, f)},accept_test=mybounds) 
            res = ls(self.err,res.x,args=(v, f)) 
            voffset, vperiod, ejs, ec, d = res.x
            space = self.errspace(self.f01,res.x,{'x':v,'y':f})
            if np.max(space) > 0.001:
                v = v[space<0.001]
                f = f[space<0.001]
                p0 = res.x
                # print(len(v),(space<0.001))
            else:
                return f, v, voffset, vperiod, ejs, ec, d
        # return f, v, voffset, vperiod, ejs, ec, d

################################################################################
### 拟合腔频调制曲线
################################################################################

class Cavitymodulation_Fit(Spec2d_Fit):

    def __init__(self,peak=15):
        self.peak = peak

    def func(self,x,paras):
        voffset, vperiod, ejs, ec, d, g, fc = paras
        tmp = np.pi*(x-voffset)/vperiod
        f01 = np.sqrt(8*ejs*ec*np.abs(np.cos(tmp))*np.sqrt(1+d**2*np.tan(tmp)**2))-ec
        fr = (fc+f01+np.sqrt(4*g**2+(f01-fc)**2))/2
        # fr = fc - g**2/(f01-fc)
        return fr

    def err(self,paras,x,y):
        return np.sum((self.func(x,paras) - y)**2)

    def fitCavitymodulation(self,v,f,s,classify=False):
        v,f = self.manipulation(v,f,s)
        paras, func = self.fitCos(v,f)
        A, C, W, phi = paras.x
        voffset, vperiod, ec, d= self.firstMax(v,f,num=0), 1/W, 0.1*np.min(f), 1
        # g = np.min(f)-fc
        ejs = (np.max(f)+ec)**2/8/ec
        g, fc = ec, np.mean(f)
        p0 = [voffset, vperiod, ejs, ec, d, g, fc]
        print(p0)
        mybounds = MyBounds(xmin=[-0.25*vperiod,0,0,0,0,0,0],xmax=[0.25*vperiod,1.5*vperiod,2*ejs,2*ec,2,2*g,2*fc])
        res = bh(self.err,p0,niter = 200,minimizer_kwargs={"method":"Nelder-Mead","args":(v, f)},accept_test=mybounds)
        # res = ls(self.err,res.x,args=(v, f)) 
        # A, C, W, phi = res.x
        voffset, vperiod, ejs, ec, d, g, fc = res.x
        return f, v, res, self.func

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

# class TwoExp_Fit(Exp_Fit):
#     def __init__(self,funcname=None,percent=0.2):
#         Exp_Fit.__init__(self,funcname)
#         self.percent = percent
#     def err(self,paras,x,y):
#         a, b, c, d, e = paras
#         return np.sum((a*np.exp(b*x) + c*np.exp(d*x) + e - y)**2)
#     def guess(self,x,y):
#         a,e,b = self.fitExp(x,y)
#         b *= -1
#         e = np.min(y) if a > 0 else np.max(y)
#         return a,b,a*self.percent,b*self.percent,e
#     def fitTwoexp(self,x,y):
#         p0 = self.guess(x,y)
#         a, b, c, d, e = p0
#         lower = [0.95*i if i > 0 else 1.05*i for i in p0]
#         higher = [1.05*i if i > 0 else 0.95*i for i in p0]
#         lower[2], lower[3] = -np.abs(a)*self.percent, -np.abs(b)*self.percent
#         higher[2], higher[3] = self.percent*np.abs(a), self.percent*np.abs(b)
#         print(p0)
#         # res = ls(self.err,p0,args=(x,y),bounds=(lower,higher))
#         res = bh(self.err,p0,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)})

#         return res.x

class TwoExp_Fit(Exp_Fit):
    def __init__(self,funcname=None,percent=0.2):
        Exp_Fit.__init__(self,funcname)
        self.percent = percent
    def fitfunc(self,x,p):
        return (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*x[None,:]), axis=0))
    def err(self,paras,x,y):
        return np.sum(((self.fitfunc(x,paras) - y)*(1.0+0.5*scipy.special.erf(0.4*(x-paras[2])))**5)**2)
    def guess(self,x,y,paras):
        offset = np.min(y)
        alist = np.max(y) - np.min(y)
        blist = np.max(x)-np.min(x)
        paras[0] = offset
        paras[1::2] = alist
        paras[2::2] = 1/blist
        return paras
    def fitTwoexp(self,x,y,num=2):
        paras = np.zeros((2*num+1,))
        xmin, xmax = np.zeros((2*num+1,)), np.zeros((2*num+1,))
        p0 = self.guess(x,y,paras)
        xmin[0], xmax[0] = p0[0]*0.5, p0[0]*1.5
        xmin[1::2], xmin[2::2] = p0[1::2]*0.5, -(np.max(x)-np.min(x))*2
        xmax[1::2], xmax[2::2] = p0[1::2]*1.5, (np.max(x)-np.min(x))*2
        mybounds = MyBounds(xmin=xmin,xmax=xmax)
        res = bh(self.err,p0,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)})
        # res = bh(self.err,res.x,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)},accept_test=mybounds)
        p0 = res.x
        print(xmin)
        # res = ls(self.err, p0, args=(np.array(x), np.array(y)))  
        return res.x

################################################################################
## 误差函数拟合
################################################################################

class Erf_fit(RowToRipe):
    def __init__(self):
        RowToRipe.__init__(self)
    def func(self,x,paras):
        sigma1, sigma2, center1, center2, a, b = paras
        return a*(scipy.special.erf((x-center1)/sigma1)+np.abs(scipy.special.erf((x-center2)/sigma2)-1))+b
    def err(self,paras,x,y):
        return np.sum((y-self.func(x,paras))**2)
    def guess(self,x,y):
        height = np.max(y) - np.min(y)
        mask = x[y < (np.max(y)+np.min(y))/2]
        center1, center2 = mask[-1], mask[0]
        b = np.mean(y - signal.detrend(y))
        a = np.max(y) - np.min(y)
        z, ynew = x[(np.min(y)+0.1*height)<y], y[(np.min(y)+0.1*height)<y]
        z = z[ynew<(np.max(ynew)-0.1*height)]
        sigma2 = (z[z<np.mean(z)][-1]-z[z<np.mean(z)][0])
        sigma1 = (z[z>np.mean(z)][-1]-z[z>np.mean(z)][0])
        return sigma1, sigma2, center1, center2, a, b
    def fitErf(self,x,y):
    
        paras = self.guess(x,y)
        # print(paras)
        res = bh(self.err,paras,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)}) 
        return res, self.func

################################################################################
## 拟合单个误差函数
################################################################################

class singleErf_fit(RowToRipe):
    def __init__(self):
        RowToRipe.__init__(self)
    def func(self,x,paras):
        sigma1, center1, a, b = paras
        return a*scipy.special.erf((x-center1)/sigma1)+b
    def err(self,paras,x,y):
        return np.sum((y-self.func(x,paras))**2)
    def guess(self,x,y):
        mask = x[y < y.mean()]
        center1 = mask[-1]
        b = np.mean(y - signal.detrend(y))
        a = np.max(y) - np.min(y)
        z = np.abs(y - np.mean(y))
        xnew = x[z<(np.max(z)+np.min(z))/2]
        sigma1 = xnew[-1] - xnew[0]
        return sigma1, center1, a, b
    def fitErf(self,x,y):

        paras = self.guess(x,y)
        # print(paras)
        res = bh(self.err,paras,niter = 50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)}) 
        return res, self.func

################################################################################
## 真空拉比拟合
################################################################################

class Vcrabi_fit():
    def __init__(self):
        pass
    def err(self,paras,x,y):
        g, A0, Z0 = paras
        return np.sum((np.sqrt(4*(g)**2+A0**2*(x-Z0)**2)-y)**2)
    def guess(self,x,y):
        Z0 = x[np.argmin(y)]
        g = np.min(y)/2
        x, y = x[x!=Z0], y[x!=Z0]
        A0 = np.mean(np.sqrt(y**2-4*(g)**2)/(x-Z0))
        return g, A0, Z0
    def fitVcrabi(self,x,y):
        p0 = self.guess(x,y)
        mybounds = MyBounds(xmin=[0,0,-1.1],xmax=[0.1,np.inf,1.1])
        res = bh(self.err,p0,niter=50,minimizer_kwargs={"method":"Nelder-Mead","args":(x, y)},accept_test=mybounds)
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


################################################################################
## 执行拟合
################################################################################

def exeFit(measure,title,data,args):
    qname = measure.qubitToread
    if title == 'singlespec':
        f_ss, s_ss = data[0], np.abs(data[1])
        index = np.abs(s_ss).argmax(axis=0)
        x,y= f_ss, s_ss
        f_rabi = np.array([x[:,i][j] for i, j in enumerate(index)])
        peak = np.array([y[:,i][j] for i, j in enumerate(index)])
        # f_rabi, peak = [], []
        # for i in range(np.shape(s_ss)[1]):
        #     index, prominences, widths = RowToRipe().findPeaks(np.abs(y[:,i]))
        #     f_rabi.append(x[:,i][index])
        #     peak.append(np.abs(y[:,i])[index])
        plt.figure()
        plt.plot(x,np.abs(y),'-')
        plt.plot(np.array(f_rabi).flatten(),np.array(peak).flatten(),'o')
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(''.join((qname[0],'_',title))))
        plt.close()
        return {j: {'f_ex':f_rabi[i]} for i, j in enumerate(qname)}

    if title == 'rabi_seq':
        v_rp, s_rp = data[0], np.abs(data[1])
        t_op, t_fit, peak = [], [], []
        for i in range(np.shape(s_rp)[1]):
            x, y = v_rp[:,i], np.abs(s_rp[:,i])
            t = RowToRipe().firstMax(x,y,num=0,peakpercent=0.8)
            peak.append(dt.nearest(x,t,y)[1])
            t_op.append(t)
        plt.figure()
        plt.plot(v_rp,s_rp,'-')
        plt.plot(t_op,peak,'o')
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(''.join((qname[0],'_',title))))
        plt.close()
        return {j: {'amp':t_op[i]} for i, j in enumerate(qname)}

    if title == 'Ramsey_seq':
    
        t_ram, s_ram = data[0], np.abs(data[1])
        x, y = t_ram[:,0], s_ram[:,0]
        res, func = T2_Fit(T1=30000,funcname='gauss',envelopemethod='hilbert').fitT2(x,np.abs(y))
        A,B,T1,T2,w,phi = res.x
        z = func(x,res.x)
        z_env = A*np.exp(-(x/T2)**2-x/T1/2) + B
        w,yt,amp = RowToRipe().fourier(x,y)
        fig, axes = plt.subplots(ncols=2,nrows=1,figsize=(9,3))
        axes[0].plot(t_ram,np.abs(s_ram),'-o',markersize=3)
        axes[0].plot(x,z)
        axes[0].plot(x,z_env)
        axes[0].set_title('$T_{2}^{*}=%.2fns,\omega=%.2fMHz$'%(T2,w*1e3))
        axes[1].plot(yt[yt!=0],np.abs(amp[yt!=0]))
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(''.join((qname[0],'_',title))))
        plt.close()
        delta = 2e6-w*1e6
        return {j: {'f_ex':measure.qubits[j].f_ex+delta} for i, j in enumerate(qname)}

    if title == 'singleZpulse':
        qubit = measure.qubits[qname[0]]
        t_shift, s_z = data
        plt.figure()
        for i in range(np.shape(s_z)[1]):
            x, y = t_shift[:,i], np.abs(s_z[:,i])
            y0 = RowToRipe().smooth(y,f0=0.2)
            res, func = Erf_fit().fitErf(x, y0)
            paras = res.x
            loc = (paras[2]+paras[3])/2
            plt.plot(x,y,'-o',markersize=3)
            plt.plot(x,func(x,paras))
            plt.vlines(loc,np.min(y),np.max(y)) 
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(''.join((qname[0],'_',title))))
        plt.close()
        zbigxy = loc-3000
        qubit.timing['z>xy'] = (zbigxy*1e-9)
        return {j: {'timing':qubit.timing} for i, j in enumerate(qname)}

    if title == 'qqTiming':
        qubit = measure.qubits[args['dcstate'][0]]
        t_shift, s_z = data
        plt.figure()
        for i in range(np.shape(s_z)[1]):
            x, y = t_shift[:,i], np.abs(s_z[:,i])
            y0 = RowToRipe().smooth(y,f0=0.2)
            res, func = singleErf_fit().fitErf(x, y0)
            paras = res.x
            loc = paras[1]
            plt.plot(x,y,'-o',markersize=3)
            plt.plot(x,func(x,paras))
            plt.vlines(loc,np.min(y),np.max(y)) 
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(''.join((args['dcstate'][0],'_',title))))
        plt.close()
        zbigxy = 1000-loc
        print(zbigxy)
        qubit.timing['read>xy'] -= (zbigxy*1e-9)
        return {j: {'timing':qubit.timing} for i, j in enumerate(args['dcstate'])}

