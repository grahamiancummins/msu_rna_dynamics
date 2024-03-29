#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
plt.rc('font', size=20)

class PPi(object):
	def __init__(self, ode, var, fig = 1):
		'''
		Produces a graphics object that acts rather like xppaut's
		"InitConds Mice". "ode" should be a models.ODE instance.  Var
		is a list of 1 or two integers.  These are indexes into the
		"ics" list passed to ode on creation (or alternately, into the
		state array of the "derivs" function passed to ode, which is
		the same thing). If there is one of these, the resulting plot
		will show this ODE state variable in Y, versus timen in X. If
		there are two, then the plot will show a phase plane cross
		section with the first referenced state on X, and the second
		on Y.

		Left clicking on the graph will integrate the ode with the 
		initial values of the variables set to the location of the 
		mouse click (In the case of a State(time) plot, only the Y
		coordinate - the state - is used. Time always starts at 0).

		Right clicking clears the graph (which will accumulate more 
		trajectory plots as you left click).

		'''
		self.ode = ode
		self.fig = plt.figure(fig)
		self.fig.add_subplot(111)
		self.ax = self.fig.get_axes()[0]
		self.ax.clear()
		self.var = var
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		if len(self.var) == 1:
			self.range(0, tt, 0, 3)
		else:
			self.range(0, 3, 0, 3)

	def range(self, xmin, xmax, ymin, ymax):
		self.ax.set_ylim([ymin, ymax])
		self.ax.set_xlim([xmin, xmax])
		self.fig.canvas.draw()

	def onclick(self, event):
		if event.button == 1:
			self.run(event.xdata, event.ydata)
		else:
			self.ax.clear()
			self.fig.canvas.draw()

	def run(self, x, y):
		if len(self.var) == 1:
			self.ode.ics[self.var[0]]=y
		else:
			self.ode.ics[self.var[0]] = x
			self.ode.ics[self.var[1]] = y
		v = self.ode.run()
		if len(self.var) == 1:
			self.ax.plot(self.ode.time, v[:,self.var[0]])
		else:
			self.ax.plot(v[:,self.var[0]], v[:,self.var[1]])
		self.fig.canvas.draw()

def scan2square(o, x=0, y=1):
	'''
	scn is the result of model.Scanner.run(True), where the 
	"stat" function used returned a scalar.
	x and y are two columns of the space to plot.

	'''
	x = np.unique(o[:,x])
	y = np.unique(o[:,y])
	s = np.zeros((x.shape[0], y.shape[0]))
	for i in range(o.shape[0]):
		s.flat[i] = o[i, -1]
	return s

def showscan(o, x=0, y=1, vnames=None, title=None, 
             save=None, fig=1, cmax=4.5, interp='nearest'):
	o = np.column_stack([ o[0], np.array(o[1])])
	i = scan2square(o, x, y)
	ext = [o[:,0].min(), o[:,0].max(), o[:,1].min(), o[:,1].max()]
	f = plt.figure(fig)
	plt.clf()
	if cmax is None:
		cmax = i.max()
	plt.imshow(i[::-1,:], aspect=1, extent=ext, 
	           vmin=0, vmax=cmax, interpolation=interp)
	if vnames:
		plt.xlabel(vnames[0])
		plt.ylabel(vnames[1])
	if title:
		plt.title(title)
	f.canvas.draw()
	if save:
		plt.savefig(save, format='png')

def splitscans(s, c):
	'''
	return two lists, the values of the scan column "c", and the 
	subsets of the scan where c has this value
	'''
	vals = np.unique(s[0][:,c])
	scans = []
	stat = np.array(s[1])
	for v in vals:
		ind = np.nonzero(s[0][:,c]== v)[0]
		ss = (s[0][ind,:], stat[ind])
		scans.append(ss)
	return (vals, scans)

def animscan(s, dname='parscan', plx=0, ply=1, frames=2, 
             vnames=('alpha2', 'epsilon', 'r')):
	'''
	Save one image for each value of the frames column of s[0]

	s is a scan result (as generated by models.parscan with a 
	scalar-valued stat function)

	'''
	if os.path.isdir(dname):
		for fn in os.listdir(dname):
			os.unlink(os.path.join(dname, fn))
	else:
		os.mkdir(dname)
	vals, scans = splitscans(s, frames)
	for i, v in enumerate(vals):
		fn = os.path.join(dname, "f%04d.png" % i)
		showscan(scans[i], plx, ply, vnames[:2], "%s = %.4g" % (vnames[2], v),
		         fn, 1)

def showBD(b, x='PAR(1)', y='PAR(2)', t = '', fig=1, clf=True):
	'''
	plot a simplified 2D bifurcation diagram

	'''
	f = plt.figure(fig)
	if clf:
		plt.clf()
	lc = 'b'
	for i, l in enumerate(b):
		xd = l[x]
		yd = l[y]
		if i % 2:
			plt.plot(xd, yd, color = lc, linewidth=2)
		else:
			lo = plt.plot(xd, yd, linewidth=2)
			lc = lo[0].get_color()
	plt.xlim([0, 3])
	plt.ylim([0,3])
	plt.xlabel("$\\alpha_2$")
	plt.ylabel("$\epsilon$")
	plt.title(t)
	f.canvas.draw()

def ucol(i, m):
	return plt.cm.spectral(float(i)/m)

def showBranchDetail(b, x='PAR(1)', y='PAR(2)', t = '', sc='ord', scr=None, 
                     nolab = [], offset=None):
	l = b.keys()
	if not sc in ['ord', 'stab']:
		if scr == None:
			scr = (b[sc].min(), b[sc].max())
			scr = (scr[0], scr[1]-scr[0])
		sc = l.index(sc)
	if type(x) == float:
		xconst = True
	else:
		x = l.index(x)
		xconst = False
	if type(y) == float:
		yconst = True
	else:
		yconst = False
		y = l.index(y)
	f = plt.figure(1)
	ax = plt.subplot(111)
	for i in range(len(b)):
		pt = b.getIndex(i)
		if xconst:
			xp = x
		else:    
			xp = pt['data'][x]
		if yconst:
			yp=y
		else:
			yp = pt['data'][y]
		if offset:
			yp+=offset*i
		if sc == 'ord':
			c =ucol(i, len(b)) 
		elif sc == 'stab':
			if pt['PT']<0:
				c = 'b'
			else:    
				c = 'r'
		else:
			cp = pt['data'][sc]
			c=ucol(cp-scr[0], scr[1])
		plt.plot([xp], [yp], '.', color = c, linestyle='None')
		if pt['TY number'] and pt['LAB']:
			if not  pt['TY name'] in ['No label', 'RG'] +nolab:
				ax.annotate(pt['TY name'], 
				            xy=(xp,yp),  xycoords='data',
				            xytext=(50, -30), textcoords='offset points',
				            arrowprops=dict(arrowstyle="->")
				            )
				print i, pt['TY name'], pt['TY number']
	f.canvas.draw()

def showAllDetail(z, **kw):
	for b in z:
		showBranchDetail(b, **kw)

def showvscan(s):
	'''
	Display the output of an aut.vscan 

	'''
	f = plt.figure(1)
	plt.clf()
	for l in s:
		isstab = l['isstable'][0]
		st = 0
		e = l['pts'][0,-1]
		u = l['pts'][:,1]
		a2 = l['pts'][:,0]
		y = 30*e + u
		for i in range(1, a2.shape[0]+1):
			if i == a2.shape[0]:
				sta = not isstab
			else:
				sta = l['isstable'][i]
			if not sta == isstab:
				x = a2[st:i]
				yl = y[st:i]
				st = i
				if isstab:
					plt.plot(x, yl, color = 'k',
					         linestyle='-',
					         linewidth=1)
				else:
					plt.plot(x, yl, color='y', linewidth=2)
				isstab = sta
	f.canvas.draw()

def fig2(s, frame=0, bd = None):
	f = plt.figure(1)
	plt.clf()
	fr = np.transpose(s[:,1:,frame])
	fr = fr[::-1,:]
	xl = 3.0/s.shape[0]
	plt.imshow(fr, aspect=1, extent=(0, 3, xl, 3),
	           interpolation='nearest')
	if bd:
		bd_outline(bd)  
	plt.xlim([0, 3])
	plt.ylim([0,3])
	plt.xlabel("$\\alpha_2$")
	plt.ylabel("$\epsilon$")
	if frame == 0:
		plt.title("Oscilatory region, Amplitude")
	if frame == 1:
		plt.title("Oscilatory region, Period")
	else:
		plt.title("Oscilatory region, min")
	f.canvas.draw()

def getdat(bd, inds, flip = (), pars=('PAR(1)', 'PAR(2)')):
	d = []
	for i in inds:
		dd = np.column_stack([bd[i][p] for p in pars])
		if i in flip:
			dd = dd[::-1, :]
		d.append(dd)
	return np.row_stack(d)

def show_region(bd, fn = '',larger = 0, olay=False):
	'''
	Create a figure similar to Yang/Kuznetzov figure 5a, using the bifurcation
	diagram bd. If fn is a non-empty string, save the figure to a file named 
	fn+'.png' as a png image.

	'''
	fig = plt.figure(1)
	if not olay:
		plt.clf()
		lcol = 'k'
	else:
		lcol = 'w'
	lfa2 = bd[0]['PAR(1)'][0]
	rfa2 = bd[2]['PAR(1)'][0]
	llhb = getdat(bd, [5, 4], [5])
	uhb = getdat(bd, [7, 6], [7])
	sp = np.argmax(uhb[:,0])
	uhbu, uhbl = uhb[:sp, :], uhb[sp:, :]
	if larger==1:
		hc = getdat(bd, [8], [])
	elif larger==None:
		hc = getdat(bd, [8, 9], [])
		hc = hc[:np.argmin(hc[:,0]), :]
	else:
		hc = getdat(bd, [9, 8], [9])
		hc = hc[:np.argmin(hc[:,0]), :]
	rf = getdat(bd, [3,2], [3])
	lf = getdat(bd, [1,0], [1])
	sp = hc[-1, 1]
	sp = np.nonzero(lf[:,1]>sp)[0]
	llf, ulf = lf[:sp,:], lf[sp:,:] 
	plt.plot(llhb[:,0], llhb[:,1], color = lcol, linestyle='-', linewidth=3)
	plt.plot(uhbu[:,0], uhbu[:,1], color = lcol, linestyle='-', linewidth=3)
	plt.plot(uhbl[:,0], uhbl[:,1], color = lcol, linestyle='--', linewidth=1)
	plt.plot(hc[:,0], hc[:,1], color = lcol, linestyle=':', linewidth=3)
	plt.plot(rf[:,0], rf[:,1], color = lcol, linestyle='--', linewidth=1)
	plt.plot(llf[:,0], llf[:,1], color = lcol, linestyle='-', linewidth=3)
	plt.plot(ulf[:,0], ulf[:,1], color = lcol, linestyle='--', linewidth=1)
	xv = np.linspace(0, rfa2, 200)
	lb = np.row_stack([llhb, np.array([[lfa2-.0001, 0]]), hc[::-1,:]])
	lb = np.interp(xv, lb[:,0], lb[:,1])
	if uhbu[:,0].min()>.1:
		ub = np.row_stack([np.array([[0, 3.0]]), uhbu])
	else:
		ub = uhbu
	ub = np.interp(xv, ub[:,0], ub[:,1])
	if not olay:
		plt.fill_between(xv, ub, lb, facecolor='y')
	plt.xlim([0, 3])
	plt.ylim([0,3])
	plt.xlabel("$\\alpha_2$", size=28)
	plt.ylabel("$\epsilon$",size=28)
	plt.xticks([0,1,2,3])
	plt.yticks([0,1,2,3])
##	try:
##		plt.title("Oscilatory region for $R=%.3G, R_m=%.3G$" % (bd[0]['PAR(3)'][0],bd[0]['PAR(4)'][0]))
##	except:
##		raise
##		plt.title("Oscilatory region for 3D model")
	fig.canvas.draw()
	if fn:
		savefig(fn + '.png')

