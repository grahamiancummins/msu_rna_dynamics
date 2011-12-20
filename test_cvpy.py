import numpy as np
import os, sys
VER=sys.version.split()[0]
VER = '.'.join(VER.split('.')[:2])
from distutils.util import get_platform

def soname(base):
    dn = "build/lib.%s-%s/%s.so" % (get_platform(), VER, base)
    return dn

def buildnload(base):
    os.system('rm %s.so' % base)
    os.system('rm -rf build')
    os.system('python setup_%s.py build | grep error' % base)
    os.system('cp %s .' % (soname(base)))
    exec("import %s as mod" % base)
    return mod

cm = buildnload('cvpy')
print cm.__file__
print dir(cm)

ic = (2.0, 1.0, .1)
par = (2.0, .1)
time = np.linspace(0, 10, 20)
o = cm.run(ic, par, time, 0)

print o

ic = (2.0, 0.0, 1.0, 0.0, .1)
par = (2.0, .1, 1.0, 1.0,2.0)
time = np.linspace(0, 10, 20)
o = cm.run(ic, par, time, 1)

print o
