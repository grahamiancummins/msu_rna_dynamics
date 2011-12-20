#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on 

# Copyright (C) 2011 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#

# from __future__ import print_function, unicode_literals
# Auto does not treat unicode as string.
import models, os, sys
import numpy as np
from operator import __add__
#import plots # - this requires editing AUTO to use the WxAgg backend
APATH = os.path.join(os.environ['HOME'], 'bin', 'auto', 'python')
sys.path.insert(0, APATH)
import auto
PDIR = os.path.join(os.environ['HOME'], 'code', 'msu')
MDIRS = ['kuz', 'rna']

def pars2silly(pars):
	d = {}
	for i, p in enumerate(pars):
		d[i+1] = p
	return d

def start(ic=None, pars=None, rna=0, tt=800.0, **kwargs):
	if not pars:
		pars = models._defaults[rna][1]
	s = models.makeode( ic, pars, tt=tt, rna = rna, run=True)
	pars = pars2silly(pars)
	tdir = os.path.join(PDIR, MDIRS[rna])
	os.chdir(tdir)
	sol = auto.load(s[-1,:], PAR=pars)
	r = auto.load(sol, e=MDIRS[rna], c=MDIRS[rna], PAR=pars, **kwargs)
	return r

def simplifyBD(bd, keeplab=False):
	'''
	reduce and AUTO bifDiag instance to something sensible.

	'''
	br = []
	for i, b in enumerate(bd.data):
		#print i
		#bd.data is a list of bifDiagBranch instances
		v = {}
		v['coords'] = b.coordnames
		v['isstable'] = []
		v['pts'] = []
		v['labs'] = {}
		for i in range(len(b)):
			pt = b.getIndex(i)
			v['isstable'].append(pt['PT']<0)
			v['pts'].append(pt['data'])
			if keeplab and (pt['TY number'] or pt['LAB']) :
				v['labs'][i] = (pt['TY name'], pt['TY number'], pt['LAB'])
			elif pt['TY number'] and pt['LAB']:
				v['labs'][i] = (pt['TY name'], pt['TY number'], pt['LAB'])
		v['pts'] = np.array(v['pts'])
		br.append(v)
	return br

def vscan1d(rna=0, r=1, apr=[0, 3], er=[0,3], n=20):
	'''
	1D continuation in alpha2 (across apr) for n values of epsilon covering er.
	rna specifies which model. If rna is True, r specifies the value of the
	parameter r (a=1, and r is not used if rna=0)

	NOTE: although it seems OK to use epsilon=0 here, it is in general NOT
	OK in AUTO. For example UZRSTOP of 0 in epsilon never triggers, and 
	there are huge numbers of branches for negative epsilon
	'''
	bd1 = []
	ic = (2.0, 2.0, 2.0)
	pars = [apr[1], er[0]]
	if rna:
		ic = ic + (2.0, 2.0)
		pars.append(r)
	for e in np.linspace(er[0], er[1], n):
		pars[1] = e
		r = start(ic, pars, rna)
		bdr = r.run()
		bd1.append(bdr)
	#for some reason simplify in this context will always fail
	#by returning None for pt['PT'] (and thus deciding istable=True
	#by returning the result and simplifying later the correct 
	#result obtains.
	return bd1

def _2parFB(pt, npts=2000):
	fw = auto.run(auto.load(pt, ISW=2, DS=.001, UZSTOP={1:[0,4], 2:[.001,4]}, NMX=npts))
	bw = auto.run(auto.load(pt, ISW=2, DS=-.001, UZSTOP={1:[0,4], 2:[.001,4]}, NMX=npts))
	return [fw, bw]

def full2DbdYK():
	b1 = auto.run(start((2,2,2), (3,.05)))
	td = _2parFB(b1('LP1'))
	td.extend(_2parFB(b1('LP2')))
	td.extend(_2parFB(b1('HB1')))
	b2 = auto.run(start((2,2,2), (3,2.5)))
	td.extend( _2parFB(b2('HB1')))
	b3 = auto.run(start((2,2,2), (3,.99)))
	periodic = auto.run(b3("HB1"),IPS=2,ICP=[1,11],NMX=200,DS=0.01,DSMAX=0.01,UZR={-11:35})
	homoclinicR =auto.run(periodic,IPS=9,ICP=[1,2],NPR=60,STOP=['BP1'],ISTART=4)
	homoclinicL =auto.run(periodic,IPS=9,ICP=[1,2],NPR=60,DS=-.01,NMX=800,STOP=['BP1'],ISTART=4)
	td.extend([homoclinicR, homoclinicL])
	td = reduce(__add__, td[1:], td[0])
	td = auto.relabel(td)
	return td

def stptsRNA():
	b1 = auto.run(start((2,2,2,2,2), (3,.05,2000), 1))
	b2 =  auto.run(start((2,2,2,2,2), (3,2.5, 2000), 1))
	return (b1('LP1'), b1('LP2'), b1('HB1'), b2('HB1'))

def mpt(b, v = 'PAR(1)'):
	i = -1
	mv = -np.inf
	vi = b.keys().index(v)
	for j in range(len(b)):
		v = b.getIndex(j)['data'][vi]
		if v > mv:
			i=j
			mv = v
	return b.getIndex(i)

def testHC(r, cc = .99):
	b3 = auto.run(start((2,2,2,2,2), (3,.99, r), 1))
	periodic = auto.run(b3('HB1'),IPS=2,ICP=[11, 1,2],NMX=200,DS=0.001,DSMAX=0.01,UZR={-11:35})
	homoclinicR =auto.run(periodic,IPS=9,ICP=[1,2,3],NPR=60,DS=.01, NMX=800, STOP=['BP1', 'RG1', "LP1"],ISTART=4)
	homoclinicL =auto.run(periodic,IPS=9,ICP=[1,2,3],NPR=60,DS=-.01,NMX=800,STOP=['BP1', 'RG1'],ISTART=4)

	return (reduce(__add__, [homoclinicR, homoclinicL], b3), periodic)

def scanforHB(er = (2,0, -.1), r = 2000):
	ic = (2,2,2,2,2)
	hbs = []
	hbis = []
	hb1 = None
	for e in apply(np.arange, er):
		b = auto.run(start(ic, (3, e, r), 1), ICP=[1,2])
		h = b("HB")
		if h:
			hbs.append( (h[0].PAR['PAR(1)'], e) )
			if not hb1:
				hb1 = h[0]
			hbis.append(h)
	hbc = auto.run(auto.load(hb1, ISW=2, DS=.001, UZSTOP={1:[0,4], 2:[.001,4]}))
	return (np.array(hbs), hbc, hbis)

def contR1():
	ic = (2,2,2,2,2)
	b = auto.run(start(ic, (3, 2.0, 200), 1), ICP=[1,2])
	l1 = auto.run(auto.load(b('LP1'), ISP=2, SP=["HB"], ISW=2, ICP=[2,1], DS=-.001, UZSTOP={1:[0,4], 2:[.001,4]}))
	l2 = auto.run(auto.load(b('LP2'), ISP=2, SP=["HB"], ISW=2, ICP=[2,1], DS=-.001, UZSTOP={1:[0,4], 2:[.001,4]}))
	hb = auto.run(auto.load(b('HB1'), ISP=2, SP=["LP2"], ISW=2, DS=.001, UZSTOP={1:[0,4], 2:[.001,4]}))
	return (b, l1, l2, hb) 

def hbs(r):
	b = auto.run(start((2,2,2,2,2), (3,.05,r), 1))
	hbl = _2parFB(b('HB1'))
	hbl = np.row_stack( [np.column_stack([hbl[0]['PAR(1)'], 
		                                  hbl[0]['PAR(2)']])[::-1,:],
		                 np.column_stack([hbl[1]['PAR(1)'], 
		                                  hbl[1]['PAR(2)']])])
	b = auto.run(start((2,2,2,2,2), (3,2.5,r), 1))
	hbh = _2parFB(b('HB1'))
	hbh = np.row_stack( [np.column_stack([hbh[0]['PAR(1)'], 
		                                  hbh[0]['PAR(2)']])[::-1,:],
		                 np.column_stack([hbh[1]['PAR(1)'], 
		                                  hbh[1]['PAR(2)']])])
	return (hbl, hbh)

def full2DbdRNA(r):
	b1 = auto.run(start((2,2,2,2,2), (3,.05,r), 1))
	td = _2parFB(b1('LP1'))
	td.extend(_2parFB(b1('LP2')))
	td.extend(_2parFB(b1('HB1')))
	b2 =  auto.run(start((2,2,2,2,2), (3,2.5,r), 1))
	td.extend( _2parFB(b2('HB1')))
	sp = mpt(td[-2][0])
	a = sp['data'][td[-1][0].keys().index('PAR(2)')]
	a = .89
	b3 = auto.run(start((2,2,2,2,2), (3,a+.1, r), 1))
	periodic = auto.run(b3('HB1'), IPS=2, ICP=[1, 11], NMX=200, 
		                DS=-0.001, DSMAX=0.01, UZR={-11:35})
	homoclinicR =auto.run(periodic, IPS=9, ICP=[1,2], NPR=60, 
		                  NMX=800, STOP=['BP1', 'RG1', "LP1"], ISTART=4)
	homoclinicL =auto.run(periodic, IPS=9, ICP=[1,2], NPR=60, DS=-.01,
		                  NMX=800, STOP=['BP1', 'RG1'], ISTART=4)
	td.extend([homoclinicR, homoclinicL])
	#return td
	#so far these are MXing. 
	td = reduce(__add__, td[1:], td[0])
	td = auto.relabel(td)
	return td

def full2DbdRNA_smallr(r):
	#still not quite right - the periodic orbit stop 
	#conditions need to be improved
	b1 = auto.run(start((2,2,2,2,2), (3,.05,r), 1))
	td = _2parFB(b1('LP1'))
	td.extend(_2parFB(b1('LP2')))
	td.extend(_2parFB(b1('HB1')))
	b2 =  auto.run(start((2,2,2,2,2), (3,2.5,r), 1))
	td.extend( _2parFB(b2('HB1')))
	hi = scanforHB(r=r)[2]
	p = auto.run(hi[-1][0], IPS=2, ICP=[1,2,11], NMX=1000, 
		         UZSTOP={1:2.35}, DS=-.001, DSMAX=0.01)
	p2 = auto.run(p, IPS=2, ICP=[2,11], NMX=100, DS=-.001,
		          STOP=['MX1'])
	homoclinicR = auto.run(p2, IPS=9, ICP=[1,2], NPR=60, 
		                   DS=-.01, NMX=800, ISTART=4)
	homoclinicL = auto.run(p2, IPS=9, ICP=[1,2], NPR=60,
		                   DS=.01, NMX=280, ISTART=4)
	td.extend([homoclinicR, homoclinicL])
	#return td
	#so far these are MXing. 
	td = reduce(__add__, td[1:], td[0])
	td = auto.relabel(td)
	return td

def full2DbdRNA_RM(r_m):
	b1 = auto.run(start((2,2,2,2,2), (3,.05,1.0, r_m), 1, tt=800))
	td = _2parFB(b1('LP1'))
	td.extend(_2parFB(b1('LP2')))
	if r_m >200:
		npts = 10000
	else:
		npts = 2000
	td.extend(_2parFB(b1('HB1'), npts))
	b2 =  auto.run(start((2,2,2,2,2), (3,2.5,1.0, r_m), 1))
	tp = _2parFB(b2('HB1'), npts)
	td.extend( [tp[0][:1], tp[1][:1]])
	sp = mpt(td[-2][0])
	a = sp['data'][td[-1][0].keys().index('PAR(2)')]
	a = .89
	b3 = auto.run(start((2,2,2,2,2), (3,a+.1, 1.0, r_m), 1))
	periodic = auto.run(b3('HB1'), IPS=2, ICP=[1, 11], NMX=200, 
		                DS=-0.001, DSMAX=0.01, UZR={-11:35})
	homoclinicR =auto.run(periodic, IPS=9, ICP=[1,2], NPR=60, 
		                  NMX=800, STOP=['BP1', 'RG1', "LP1"], ISTART=4)
	homoclinicL =auto.run(periodic, IPS=9, ICP=[1,2], NPR=60, DS=-.01,
		                  NMX=800, STOP=['BP1', 'RG1'], ISTART=4)
	td.extend([homoclinicR, homoclinicL])
	#return td
	#so far these are MXing. 
	td = reduce(__add__, td[1:], td[0])
	td = auto.relabel(td)
	return td