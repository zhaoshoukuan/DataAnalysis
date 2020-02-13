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
import serial, time, numpy as np
from scipy import fftpack
from scipy.optimize import least_squares as ls, curve_fit
import asyncio

################################################################################
### 设置衰减器
################################################################################

class Att_Setup():
    
    def __init__(self,com):
        self.com = com
        ser = serial.Serial(self.com,baudrate=115200, parity='N',bytesize=8, stopbits=1, timeout=1)
        self.ser = ser
        if ser.isOpen():    # make sure port is open     
            print(ser.name + ' open...')
            ser.write(b'*IDN?\n')
            x = ser.readline().decode().split('\r''\n')
            print(x[0])
            ser.write(b'ATT?\n')
            y = ser.readline().decode().split('\r''\n')
            print('ATT',y[0])

    def Att(self,att):

        self.ser.write(b'ATT %f\n'%att)
        time.sleep(1)
        self.ser.write(b'ATT?\n')

    def close(self):

        self.ser.close()

################################################################################
### 收集awg波形名字
################################################################################

def Collect_Waveform(dictname,kind):
    if kind in dictname:
        raise 'kind has already existed'
    def decorator(func):
        def wrapper(*args, **kw):
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.get_event_loop()
                name_list = loop.run_until_complete(func(*args, **kw))
                dictname[kind] = name_list
            else:
                return func(*args, **kw)
        return wrapper
    return decorator

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

################################################################################
### 拟合Exp函数
################################################################################

class Exp_Fit(RowToRipe):

    def __init__(self,funcname=None):
        self.funcname = funcname
    
    def errExp(self,paras, x, y):
        
        if self.funcname == 'gauss':
            A, B, T1, T2 = paras
            return A * np.exp(-(x/T2)**2-x/T1/2) + B - y
        else:
            A, B, T1 = paras
            return A * np.exp(-x/T1) + B - y

    def guessExp(self,x,y):
        ymin = y.min()
        y = y-y.min()
        mask = y > 0
        if self.funcname == 'gauss':
            a = np.polyfit(x[mask], np.log(y[mask]), 2)
            return [np.exp(a[2]), ymin, np.abs(1/a[1]/2), np.sqrt(1/np.abs(a[0]))]
        else:
            a = np.polyfit(x[mask], np.log(y[mask]), 1)
            return [np.exp(a[1]), ymin, np.abs(1/a[0])]

    def fitExp(self,x,s):
        y = np.abs(s)
        p0 = self.guessExp(x,y)
        res = ls(self.errExp, p0, args=(x, y)) 
        return res.x

################################################################################
### 拟合Cos函数
################################################################################

class Cos_Fit(RowToRipe):

    def __init__(self):
        pass

    def errCos(self,paras,x,y):
        A,C,W,phi = paras             
        return  A*np.cos(2*np.pi*W*x+phi)+C-y  

    def guessCos(self,x,y):
        x, y = np.array(x), np.array(y)
        sample = (np.max(x) - np.min(x))/(len(x) - 1)
        Ag, Cg= np.max(y)-np.min(y), np.mean(y) 
        yt  = np.fft.fftshift(np.fft.fftfreq(len(y))) / sample
        amp = np.fft.fftshift(np.fft.fft(y))
        z = np.abs(amp[yt!=0])
        Wg = np.abs(yt[np.argmax(z)])
        phig =  np.mean(np.arccos((y[0] - Cg)/Ag) - Wg*x[0])
        return Ag, Cg, Wg, phig

    def fitCos(self,volt,s):
        x, y = volt, np.abs(s)
        Ag, Cg, Wg, phig = self.guessCos(x,y)
        res = ls(self.errCos, [Ag,Cg,Wg,phig], args=(x, y))         
        A, C, W, phi = res.x
    
        return A, C, W, phi

################################################################################
### 拟合洛伦兹函数
################################################################################

class Lorentz_Fit(RowToRipe):
    '''
    processing x in GHz
    '''
    def __init__(self):
        pass

    def errLorentz(self,para,x,y):
        a,b,c,d = para
        return a/(1.0+c*(x-b)**2)+d-y

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
            raise 'please divided by 1e9, processing x in GHz'
        para = self.guessLorentz(x,y)
        res = ls(self.errLorentz,para,args=(x,y))
        a,b,c,d = res.x
        return a,b,c,d,np.sqrt(np.abs(a/c))*2e3

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
    def __init__(self,responsetime=100,T1=25000,funcname=None,envelopemethod=None):
        Exp_Fit.__init__(self,funcname)
        self.responsetime = responsetime
        self.T1 = T1
        self.envelopemethod = envelopemethod

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
    
    def guessT2(self,x,y_new,y):
 
        A, B, T1, T2 = self.fitExp(x[5:-10],y_new[5:-10])
        if np.abs(self.T1-T1)>5000:
            T1 = self.T1
        Ag, Cg, Wg, phig = self.guessCos(x,y)
        return A, B, T1, T2, Wg, phig

    def errT2(self,para,x,y):
        A,B,T1,T2,w,phi = para
        return A*np.exp(-(x/T2)**2-x/T1/2)*np.cos(2*np.pi*w*x+phi) + B - y

    def fitT2(self,x,y):
        d = self.space(y)
        if self.envelopemethod == 'hilbert':
            out = self.envelope_Hilbert(y)
        else:
            out = self.envelope(y)
        A,B,T1,T2,w,phi = self.guessT2(x,out,y)
        if T2 > 0.8*x[d-1] and d < 0.8*len(y):
            T2 = 0.37*x[d-1]
        p0 = A,B,T1,T2,w,0
        print(p0)
        res = ls(self.errT2, p0, args=(x, y))         
        A,B,T1,T2,w,phi = res.x
        return A,B,T1,T2,w,phi,x[:d-1],out[:d-1]

class Rabi_Fit(T2_Fit):

    def __init__(self,responsetime=100,T1=20000,funcname=None,envelope=None):
        T2_Fit.__init__(self,responsetime,T1,funcname,envelope)

    
    def guessRabi(self,x,y_new,y):
 
        A, B, T1 = self.fitExp(x[5:-10],y_new[5:-10])
        if np.abs(self.T1-T1)>5000:
            T1 = self.T1
        Ag, Cg, Wg, phig = self.guessCos(x,y)
        return A, B, T1, Wg, phig

    def errRabi(self,para,x,y):
        A,B,T1,w,phi = para
        return A*np.exp(-x/T1)*np.cos(2*np.pi*w*x+phi) + B - y

    def fitRabi(self,x,y):
        if self.envelopemethod == 'hilbert':
            out = self.envelope_Hilbert(y)
        else:
            out = self.envelope(y)
        A,B,T1,w,phi = self.guessRabi(x,out,y)
        A, B = np.max(y) - np.min(y), np.mean(y)
        p0 = A,B,T1,w,0
        res = ls(self.errRabi, p0, args=(np.array(x), np.array(y)))         
        A,B,T1,w,phi = res.x
        return A,B,T1,w,phi