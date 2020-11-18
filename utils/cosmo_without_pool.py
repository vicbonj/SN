import numpy as np
import astropy.constants as cst
from scipy import integrate
from itertools import repeat
import os, ctypes
import platform

#os.system('gcc -dynamiclib -o testlib.dylib -lm -fPIC testlib.c')
#os.system('gcc -shared -o testlib.so -lm -fPIC testlib.c')

if platform.uname()[0] != 'Darwin':
    dir1 = os.path.abspath('testlib.so')
    lib = ctypes.cdll.LoadLibrary(dir1)
    integrand_in_c = lib.f
    integrand_in_c.restype = ctypes.c_double
    integrand_in_c.argtypes = (ctypes.c_int, ctypes.c_double)
    
    dir2 = os.path.abspath('testlib_total.so')
    lib2 = ctypes.cdll.LoadLibrary(dir2)
    integrand_total_in_c = lib2.f
    integrand_total_in_c.restype = ctypes.c_double
    integrand_total_in_c.argtypes = (ctypes.c_int, ctypes.c_double)
else:
    dir1 = os.path.abspath('testlib.dylib')
    lib = ctypes.cdll.LoadLibrary(dir1)
    integrand_in_c = lib.f
    integrand_in_c.restype = ctypes.c_double
    integrand_in_c.argtypes = (ctypes.c_int, ctypes.c_double)

    dir2 = os.path.abspath('testlib_total.dylib')
    lib2 = ctypes.cdll.LoadLibrary(dir2)
    integrand_total_in_c = lib2.f
    integrand_total_in_c.restype = ctypes.c_double
    integrand_total_in_c.argtypes = (ctypes.c_int, ctypes.c_double)

def ez(z, om, ol, ok, w):
    return (om*(1+z)**3 + ok*(1+z)**2 + ol*(1+z)**(3*(1+w)))**(1/2.)

def ez_total(z, om, ol, orad, ok):
    return (orad*(1+z)**4 + om*(1+z)**3 + ok*(1+z)**2 + ol)**(1/2.)

def integrand(z, om, ol, ok, w):
    return 1/ez(z, om, ol, ok, w)

def integrand_total(z, om, ol, orad, ok):
    return 1/ez_total(z, om, ol, orad, ok)

def integr(inputs):
    z, om, ol, ok, w, in_c = inputs
    if in_c == True:
        return integrate.quad(integrand_in_c, 0, z, args=(om, ol, ok, w))[0]
    else:
        return integrate.quad(integrand, 0, z, args=(om, ol, ok, w))[0]

def integr_total(inputs):
    z, om, ol, orad, ok, in_c = inputs
    if in_c == True:
        return integrate.quad(integrand_total_in_c, 0, z, args=(om, ol, orad, ok))[0]
    else:
        return integrate.quad(integrand_total, 0, z, args=(om, ol, orad, ok))[0]
    
def dl(z, om, ol, ok, w, in_c):
    inputs = zip(z, repeat(om), repeat(ol), repeat(ok), repeat(w), repeat(in_c))
    ee = map(integr, inputs)
    return (1+z) * cst.c.to('km/s').value * list(ee)

def dl_total(z, om, ol, orad, ok):
    inputs = zip(z, repeat(om), repeat(ol), repeat(orad), repeat(ok))
    ee = map(integr_total, inputs)
    if ok > 0:
        return (1+z) * cst.c.to('km/s').value * np.sinh(np.sqrt(ok)*np.array(list(ee))) / np.sqrt(ok)
    elif ok < 0:
        return (1+z) * cst.c.to('km/s').value * np.sin(np.sqrt(abs(ok))*np.array(list(ee))) / np.sqrt(abs(ok))
    else:
        return (1+z) * cst.c.to('km/s').value * list(ee)

def mul(z, om, ol, ok, w, in_c):
    return 5*np.log10(dl(z, om, ol, ok, w, in_c)) - 5

def mul_total(z, om, ol, orad, ok):
    return 5*np.log10(dl_total(z, om, ol, orad, ok)) - 5

def tofit_total(z, om, ol):
    orad = 0
    ok = 1 - om - ol - orad
    return mul_total(z, om, ol, orad, ok)

def tofit(z, om, w, in_c):
    #ok = 1 - ol - om
    ok = 0
    ol = 1 - om
    return mul(z, om, ol, ok, w, in_c)
