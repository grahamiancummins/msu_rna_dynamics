#!/usr/bin/env python

import numpy as np 
import scipy.integrate as si
import cvpy
import os

def kuzuprime(u, t0, al2=3, ep=.05, al1=5, n=3):
	'''
	Derivatives function for the Yang/Kuznetzov (Y/K) genetic 
	oscillator model. Compatible with Scipy.integrate.odeint.
	The t0 argument is ignored (because this system is
	autonymous, but it is required in the api by scipy, since 
	not all systems are).

	U is a 3-vector, and represents the states (u,v,w) from Yang.
	The parameters are alpha2, epsilon, alpha1, and n.
	The return value is (du, dv, dw)
	'''
	u,v,w = u
	du = al1/(1+v**n) - u
	dv = al1/(1+w**n) + al2/(1+u**n) - v
	dw = ep*(al1/(1+u**n) - w)
	return (du, dv, dw)

def kuzrna(u, t0, al2 = 3, ep = 0.05, r = 1.0, r_m=1.0, a = 1.0, al1=5.0, n=3):
	'''
	Modification of the Y/K model to include RNA for the two 
	protiens (u and v).     
	a scales the rate of protien synthesis, and r scales the entire
	relative speed of protien vs RNA components.

	u is 5D = (u_rna, u, v_rna, v, w)

	'''
	u_rna, u, v_rna, v, w = u
	du_rna = r_m*(al1/(1+v**n) - u_rna) 
	du = r*(a*u_rna - u)
	dv_rna = r_m*(al1/(1+w**n) + al2/(1+u**n) - v_rna)
	dv = r*(a*v_rna-v)
	dw = ep*(al1/(1+u**n) - w)
	return (du_rna, du, dv_rna, dv, dw)

xppyk = [
    "u' = ap1/(1+v^n) - u",
    "v' = ap1/(1+w^n)+ap2/(1+u^n) - v",
    "w' = e*(ap1/(1+u^n)-w)",
    "par ap1=5,n=3"
]
xppyk_icn = ('u','v','w')
xppyk_pn = ('ap2', 'e')

xpprna = [
    "u1' = r_m*(ap1/(1+v2^n) - u1)",
    "u2' = r*(a*u1-u2)",
    "v1' = r_m*(ap1/(1+w^n)+ap2/(1+u2^n) - v1)",
    "v2' = r*(a*v1-v2)",
    "w' = e*(ap1/(1+u2^n)-w)",
    "par ap1=5,n=3,a=1,r=1,r_m=1"
]
xpprna_icn = ('u1', 'u2', 'v1', 'v2', 'w')
xpprna_pn = ('ap2', 'e', 'r', 'r_m')

class ODE(object):
	'''
	A small class that stores the parameters for integrating an
	system of odes with scipy.integrate.odeint.

	The __init__ method stores the needed information, and the
	run method excecutes and returns a NxM array. N is the 
	length of self.time, and each row corresponds to the vector
	of states at the corresponding time.

	Since this is such a tiny class, it has no accessor methods.
	The members "pars" and "ics" store the parameters and 
	initial conditions as lists. Clients should simply edit
	these directly. Time is calculated internally as an 
	numpy.linspace, so it is probably better to make a new 
	instance than to try to edit self.time.

	'''
	def __init__(self, derivs, pars, ics, tt=100.0, dt=0.1):
		'''
		Derivs is a function that caluculates derivatives 
		(as used by odeint). Pars is a tuple of additional 
		parameters (after the first two, U, and T) for this
		function. ics is a tuple, list, or array giving the 
		initial values of the states. tt is the total 
		integration time (in seconds). Dt is the time period
		between samples (note that odeint is internally a 
		variable timestep method, so dt really acts as an 
		upper bound on the time step, but also it determines 
		the sampling period of self.time, which is the array of
		times at which the solution is measured and returned.

		'''
		self.f = derivs
		self.pars = list(pars)
		self.ics = list(ics)
		self.time = np.linspace(0, tt, tt/dt)

	def run(self):
		'''
		Integrate, and return the solutions.

		'''
		return si.odeint(self.f, np.array(self.ics), self.time, tuple(self.pars))

class XPPODE(ODE):
	def __init__(self, xpp, pars, ics, parnames, icnames, tt=100.0, dt=0.1):
		'''
		xpp is a list of strings.

		'''
		self.f = xpp
		self. icn = icnames
		self.pn = parnames
		self.fname = 'py_xpp.ode'
		self.oname = 'output.dat'
		self.pars = list(pars)
		self.ics = list(ics)
		self.dt = dt
		self.tt = tt
		self.time = np.linspace(0, tt, tt/dt)

	def writexpp(self):
		xppl = self.f[:]
		p = 'par '
		for i in range(len(self.pn)):
			p= p+"%s=%.3g," % (self.pn[i], self.pars[i])
		xppl.append(p[:-1])
		init = "init "
		for i in range(len(self.icn)):
			init= init+"%s=%.3g," % (self.icn[i], self.ics[i])
		xppl.append(init[:-1])
		xppl.append("@maxstor=2000000,dt=%.3g,total=%.3g" % (self.dt, self.tt))
		xppl.append("done")
		open(self.fname, 'w').write("\n".join(xppl))

	def run(self):
		if os.path.isfile(self.oname):
			os.unlink(self.oname)
		self.writexpp()
		os.system('xppaut -silent %s' % self.fname)
		out = np.array([map(float, x.split()) for x in open(self.oname).readlines()])
		return out[:, 1:]

class CVODE(ODE):
	'''
	ODE Version that uses the cvpy sundials wrapper. For efficiency,
	this defines the derivative functions at the C level.
	Consequently, this class is less general than ODE. The first 
	argument to the constructer is not a function. Instead it is a 
	flag, used as "userna" in the call to cvpy.run. This makes this 
	class specific to this (3 element genetic oscillator) problem. 

	userna = 0 -> Y/K model, ics are a 3-vector (u, v, w), 
	              pars are a 2-vector (alpha2, epsilon) 
	              (alpha1 fixed = 5, and n fixed = 3)
	              This is similar to kuzuprime above, except that 
	              the final two parameters are fixed at the defaults.
	       = 1 -> 5D rna modification of Y/K with scaling parameter 
	              R. Similar to kuzrna above. ics are a 5-vector
	              (u, u2, v, v2, w), pars are (al2, ep, r, a) 
	              (again, alpha1 and n are fixed at (5,3))
	'''

	def run(self):
		return cvpy.run(self.ics, self.pars, self.time, self.f)

_defaults = [ ((2.0, 2.0, 2.0), (3, .05)), 
              ((2.0, 2.0, 2.0, 2.0, 2.0),(3, .05, 1.0, 1.0)) ]
# the first element is (ic, pars) for the y/k model, the
#second for the rna model
_oderhs = [kuzuprime, kuzrna]
_xpppars = [(xppyk, xppyk_pn, xppyk_icn), 
            (xpprna, xpprna_pn, xpprna_icn)]


def makeode( ic=None, pars=None, dt=.1, tt=400.0, mode ='c', rna = 0, run=False):
	'''
	Quick generation of ODE instances for various methods.
	ic -> initial conditions tuple
	pars -> parameter tuple
	tt -> total integratin time (float)
	dt -> output time step (float)
	mode -> which type of ODE to use ('s'  | 'c' | 'x')
	rna -> use the rna model? (0 | 1)
	run -> run the model? (0 | 1)

	mode "c" is the cvode c extension, 
	    "s" is simple (aka scipy odeint)
	    'x' is xppaut

	several other modes, including scipy.integrate.ode and 
	python-sundials were rejected for poor performance. 

	"c" is probably best. "s" is comparable for most parameter 
	choices, and has the fewest requirements. "x" is a dirty 
	slow hack, but is included for comparisons to earlier work.

	Although performance of odeint and sundials is comparable for
	"default" paramaters, some parameter sets, particularly high
	values of epsilon, have much worse performance in odeint.  If
	I had to guess, I'd say it is handling stiffness with reduced
	time step, compared to Newton iteration + BDF in sundials.
	Anyway, across the 0-3 range of both parameters, "c"
	outperforms "s" by ~60x

	rna 0 -> the Y/K 3D model system, ic is (u, v, w), pars is 
	        (alpha2, epsilon)
	rna 1 -> the 5D partial rna model (treating w as a small 
	        molecule). ic is (u_rna, u, v_rna, v, w), pars is 
	        (alpha2, epsilon, r, a)

	If ic or pars is ommited, the "_defaults" values get used.

	if run is true, return the result of the integration. 
	Otherwise, return the ODE instance.

	'''
	if ic == None:
		ic = _defaults[rna][0]
	if pars == None:
		pars = _defaults[rna][1]
	if mode == 'c':
		o =CVODE(rna, pars, ic, tt, dt) 
	elif mode == 's':
		o = ODE(_oderhs[rna], pars, ic, tt, dt)
	elif mode =='x':
		o = XPPODE(_xpppars[rna][0], pars, ic, 
				   _xpppars[rna][1], _xpppars[rna][2], tt, dt)
	if run:
		o = o.run()
	return o

class Scanner(object):
	'''
	Class that provides for repeated runs of an ODE instance
	with different pars and ics.

	'''
	def __init__(self, ode, pars, ics, stat):
		'''
		ode is an ODE instance.

		Pars and ics are tuples of the same length as the 
		pars and ics tuples used in the ode, but the values 
		are collections, rather than numbers. These may have 
		0, 1, or many values. The scanner will construct 
		every combination of values in these ranges. (if a 
		given index has 0 values, this is the same as having one
		value that is the one which is used at the corresponding
		index in the ODE object). 

		stat is a function which should expect as input the 
		sort of array returned by running the ode, and returns 
		some calculated value (which is hopefully much smaller, 
		since a run of the scanner may result in thousands of 
		runs of the ODE, and thus the output structure
		resulting from stat = lambda x: x is very likely to 
		overflow your RAM

		'''

		self.ode = ode
		self.pars = list(pars)
		for i, p in enumerate(self.pars):
			if len(p) == 0:
				self.pars[i] = [self.ode.pars[i]]
		self.ics = list(ics)
		for i, p in enumerate(self.ics):
			if len(p) == 0:
				self.ics[i] = [self.ode.ics[i]]
		self.stat = stat

	def combinations(self, lol):
		'''
		Take a list of lists of numbers. 
		Return an array of float64, with shape NxM where
		N = cumprod([len(x) for x in lol]) and  
		M = len(lol). 
		The return array contains all possible
		combinations of the inputs (eg, sets of items containing
		one item from each of the lists in lol)

		'''
		m=len(lol)
		shapes=[len(x) for x in lol]
		stride=np.cumproduct(shapes)
		n = stride[-1]
		oa=np.zeros((n, m), np.float64)
		oa[:,0]=np.resize(lol[0], n)
		for i in range(1, m):
			dl=stride[i-1]
			dr=n/dl
			oa[:,i]=np.ravel(np.transpose(np.resize(lol[i], (dl, dr))))    
		return oa

	def range(self):
		'''
		return an NxM array, where:
		    M = len(self.pars) + len(self.ics)
		    N is the size of the scanned space (all combinations
		    of pars and ics).
		each row in the array is a set of pars, 
		followed by a set of ics

		'''
		return self.combinations( self.pars + self.ics)

	def run(self, retrng=False):
		n = len(self.pars)
		out = []
		rng = self.range()
		import time
		for cond in rng:
			t = time.time()
			self.ode.pars = list(cond[:n])
			self.ode.ics = list(cond[n:])
			v = self.ode.run()
			out.append(self.stat(v))
		if retrng:
			return (rng, out)
		else:
			return out

def oscil(out, i=0):
	hp = int(out.shape[0]/2.0)
	out = out[hp:,i]
	return out.max() - out.min()

def parscan(ode, dt=.1, tt=400.0, ranges=((0,3,60), (0,3,60)), stat=oscil):
	'''
	construct a parameter scan for ODE instance ODE using the 
	specified integration time, step, and stat function. 

	Ranges specifies the parameter search range. It should be 
	equal to or less than the length of ode.pars. If it is less, 
	extra elements recieve the range parameter [] ( meaning that they
	keep the value that was set on the creation of ode. See Scanner
	documentation). Specified values may be one of:
	    a 3-tuple, which is used as an argument to np.linspace
	    a float, which is converted to a one-element sequence 
	        (of that float)
	    any other sequence, which is passed literally. (this may
	    include [] )
	'''
	pars = []
	for r in range(len(ode.pars)):
		if r<len(ranges):
			rv = ranges[r]
		else:
			rv = []
		if type(rv) == tuple and len(rv) == 3:
			pars.append(apply(np.linspace, rv))
		elif type(rv) in [int, float]:
			pars.append( [rv])
		else:
			pars.append( rv )
	ics = [ [] for i in range(len(ode.ics))]
	s = Scanner(ode, pars, ics, stat)
	return s.run(True)

def qscan(r = 1, npts=100, dt = .1, tt=600):
	if r < 1:
		rm = 1.0/r
		r = 1
	else:
		rm = 1
	ode = makeode((2,2,2,2,2), (2.0, 1.0, r, rm), dt, tt, 'c', 1, False)
	ranges=((0,3,npts), (0,3,npts))
	return parscan(ode, dt, tt, ranges)

def trackright(spars=(2.0, 1.0, 3, 1.0, 1.0), iic=(2,2,2,2,2), step=.05, stop = 2.7, dt = .1, tt=400.0, stat=oscil):
	ode =makeode(iic, spars, dt, tt, 'c', 1, False) 
	a2 = spars[0]
	spars = list(spars[1:])
	a2r = np.arange(a2, stop, step)
	res = []
	stats = []
	lastic = iic
	for a2 in a2r:
		ode.pars = [a2]+spars
		ode.ics = lastic
		v = ode.run()
		res.append(v)
		stats.append((a2, stat(v)))
		lastic = tuple(v[-1])
	return (np.array(stats), res)	

def _lmax(v, nr = 3):
	lm = False
	i = nr
	while not lm:
		if all(v[i]>=v[i-nr:i+nr]):
			lm = i
		elif i+nr+1>=v.shape[0]:
			lm = v.shape[0]
		else:
			i+=1
	return lm 

def _stats(v):    
	hp = int(v.shape[0]/2.0)
	v = v[hp:,0]
	ran = v.max() -v.min()
	m1 = _lmax(v)
	if m1 >= v.shape[0]-3:
		per = 0
	else:
		per = _lmax(v[m1:]) - m1
	mv = v.min()
	return (ran, per, mv)

def _trf(e, a2l, step, a2r, ode):
	a2 = a2l
	res = []
	while a2 < a2r:
		ode.pars[0] = a2
		v = ode.run()
		res.append(_stats(v))
		ode.ics = list(v[-1])
		a2 += step
	return np.array(res)

def trackright_hb(r = 1, step = .05, dt = .1, tt=400.0):
	import aut
	ode =makeode((2,2,2,2,2), (0, 0, r, 1.0, 1.0), dt, tt,'c', 1, False) 
	nsteps = 3/step
	ev = np.linspace(0, 3, nsteps)
	hl, hr = aut.hbs(r)
	a2l = np.interp(ev, hl[:,1], hl[:,0])
	a2r = np.interp(ev, hr[:,1], hr[:,0])
	scan = np.zeros((ev.shape[0],ev.shape[0], 3))
	for i in range(ev.shape[0]):
		if a2l[i]+step>=a2r[i]:
			continue
		st = np.nonzero(ev>=a2l[i])[0]-1
		ode.pars[1]=ev[i]
		v = _trf(ev[i], a2l[i], step, a2r[i], ode) 
		scan[st:st+v.shape[0], i] = v
	return scan

def rtr(spars=(2.0, .2, 3, 1.0, 1.0), iic=(2,2,2,2,2), step=(.05, .05), stop=(2.7, 2.0)):
	e=spars[1]
	spars = list(spars)
	er = np.arange(e, stop[1], step[1])
	ar = np.arange(spars[0], stop[0], step[0])
	out = np.zeros((ar.shape[0], er.shape[0]))
	for i, e in enumerate(er):
		spars[1] = e
		st, _ = trackright(spars, iic, step[0], stop[0])
		out[:,i] = st[:,1]
	return ((ar[0], ar[-1], er[0], er[-1]), out[:,::-1].transpose())


def _ev(ode, pars, ics):
	ode.pars = pars
	ode.ics = ics
	v = ode.run()
	return (oscil(v), v[-1])

def trackdr(bd, step=.005, emin=.05, dt=.1, tt=400.0, nrand=10, randsd=.1):
	lpl = bd[0]['PAR(1)'][0]
	hb = bd[-4]
	hbr = hb['PAR(1)'].max()
	hbc = np.nonzero(hb['PAR(1)']>=lpl)[0]
	pt = hb.getIndex(hbc)['data']
	ici =[hb.keys().index('U(%i)' % i) for i in range(1,6)]
	ic = [pt[i] for i in ici]
	pi = [hb.keys().index('PAR(%i)' % i) for i in range(1,4)]
	pars = [pt[i] for i in pi] + [1.0]
	pars[0] = pars[0]-10*step
	rpars = pars[2:]
	ext = [pars[0], hbr, emin, pars[1]]
	npts = [int(round((ext[i]-ext[i-1])/step)) for i in [3,1] ]
	ar = np.linspace(ext[0], ext[1], npts[1])
	er = np.linspace(ext[3], ext[2], npts[0])
	out = np.zeros(npts)
	ode = makeode(ic, pars, dt, tt, 'c', 1, False)
	lrics = [None for _ in range(npts[1])]
	for ei in range(npts[0]):
		trics = [None for _ in range(npts[1])]
		for ai in range(npts[1]):
			o, nic = _ev(ode, [ar[ai], er[ei]]+rpars, ic)
			if not o:
				tics = []
				if ai>0:
					tics.append(trics[ai-1])
					if not lrics[ai-1] is None:
						tics.append(lrics[ai-1])
				if not lrics[ai] is None:
					tics.append(lrics[ai])
				for _ in range(nrand):
					noise = np.random.normal(0, randsd, 5)
					tics.append([ic[i]+noise[i] for i in range(5)])
				for i in range(len(tics)):
					o, nic = _ev(ode, [ar[ai], er[ei]]+rpars, tics[i])
					if o:
						break
			if o:
				out[ei, ai] = o
				ic = nic
				trics[ai] = nic
			else:
				break
		lrics = trics
	return (ext, out)
