/*
Created on: May 15 by Graham Cummins

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program; if not, write to the Free Software Foundation, Inc., 59 Temple 
Place, Suite 330, Boston, MA 02111-1307 USA 2011
*/

#include <stdio.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_dense.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// NV_Ith_S(v, i) to access vector elements, DENSE_ELEM(A, i, j) for matrix

#define	KUZ_ND	3
#define AL1 RCONST(5.0)
#define ONE RCONST(1.0)
#define A_C RCONST(1.0)
#define C_ARRAY NPY_ALIGNED | NPY_CONTIGUOUS | NPY_FORCECAST


static realtype pow3(realtype x)
{
	return x*x*x;
}

static int kuz(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
	realtype * ud;
	/* u,v,w = y, du, dv, dw = ydot
		du = al1/(1+v**n) - u
	dv = al1/(1+w**n) + al2/(1+u**n) - v
	dw = ep*(al1/(1+u**n) - w)

	I declare (because Sundials itself doesn't seem to) that *user_data is
	in fact a realtype *, referencing a 2 element array, 
	alpha2, epsilon  :P
	*/
ud = (realtype *) user_data;
NV_Ith_S(ydot,0) = AL1/(RCONST(1.0) + pow3(NV_Ith_S(y, 1))) - NV_Ith_S(y, 0);
NV_Ith_S(ydot,1) = AL1/(RCONST(1.0) + pow3(NV_Ith_S(y, 2)))
	+ ud[0]/(1 + pow3(NV_Ith_S(y, 0)))
- NV_Ith_S(y, 1);
NV_Ith_S(ydot,2) = ud[1]*(AL1/(RCONST(1.0) + pow3(NV_Ith_S(y, 0))) - NV_Ith_S(y, 2));
return(0);
}

static int rna(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
	/*
	u_rna, u, v_rna, v, w = y
	ydot = du_rna, du, dv_rna, dv, dw

	userdata is realtype[5] alpha2, ep, r, r_m

	du_rna = r_m*(al1/(1+v**n) - u_rna) 
	du = r*(a*u_rna - u)
	dv_rna = r_m*(al1/(1+w**n) + al2/(1+u**n) - v_rna)
	dv = r*(a*v_rna-v)
	dw = ep*(al1/(1+u**n) - w)
	*/
	realtype * ud;
	ud = (realtype *) user_data;
	NV_Ith_S(ydot,0) = ud[3]*(AL1/(ONE + pow3(NV_Ith_S(y, 3))) 
		- NV_Ith_S(y, 0));
	NV_Ith_S(ydot,1) = ud[2]*(A_C*NV_Ith_S(y, 0) 
		- NV_Ith_S(y, 1));
	NV_Ith_S(ydot,2) = ud[3]*(AL1/(ONE + pow3(NV_Ith_S(y, 4)))
		+ ud[0]/(ONE + pow3(NV_Ith_S(y, 1)))
		- NV_Ith_S(y, 2));
	NV_Ith_S(ydot,3) = ud[2]*(A_C*NV_Ith_S(y, 2) 
		- NV_Ith_S(y, 3));
	NV_Ith_S(ydot,4) = ud[1]*(AL1/(ONE + pow3(NV_Ith_S(y, 1))) 
		- NV_Ith_S(y, 4));
	return(0);	
}

/*

static int Jac(int N, realtype t,
	N_Vector y, N_Vector fy, DlsMat J, void *user_data,
	N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

But we don't know Jac
*/


static realtype * packpytup (PyObject * tup) 
{
	int i, n;
	realtype * a;
	n = (int) PySequence_Length(tup);
	a = (realtype *) malloc( sizeof(realtype)*n);
	for (i=0;i<n;i++) {
		a[i] = (realtype) PyFloat_AsDouble(PySequence_GetItem(tup, i)); 
	}
	return a;
}

static PyObject *
giccv_run(PyObject *self, PyObject *args)
{
	PyObject *icpy, *parpy;
	PyArrayObject *times, *output;
	N_Vector y;
	void *cvode_mem;
	int i, j, userna, flag;
	long rshape[2];
	realtype * par;
	realtype * ic;
	realtype t, next;
	if (!PyArg_ParseTuple(args, "OOOi", &icpy, &parpy, &times, &userna))
		return NULL;
	rshape[0] = (long) times->dimensions[0];
	rshape[1] = (long) PySequence_Length(icpy);
	output = PyArray_ZEROS(2, rshape, NPY_FLOAT64, 0);
	//Py_INCREF(output);
	//This is wrong. PyArray_ .... must incref already
	// with this line, it leaks memory like crazy
	ic = packpytup(icpy);
	par = packpytup(parpy);
	y = N_VMake_Serial(rshape[1], ic);
	cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);

	flag = CVodeSetUserData(cvode_mem, par);
	if (userna == 0) {
		flag = CVodeInit(cvode_mem, kuz, RCONST(0.0), y);
	} else {
		flag = CVodeInit(cvode_mem, rna, RCONST(0.0), y);
	}
	flag = CVodeSStolerances(cvode_mem, RCONST(1e-6), RCONST(1e-6));
		flag = CVDense(cvode_mem, rshape[1]);
	t = *(realtype *) PyArray_GETPTR1(times,0);
	for (j=0;j<rshape[1];j++) {
		*(double *)PyArray_GETPTR2(output, 0, j) = (double)
		NV_Ith_S(y, j);
	}
	for (i=1;i<rshape[0];i++) {
		next = *(realtype *) PyArray_GETPTR1(times,i);
		flag = CVode(cvode_mem, next, y, &t, CV_NORMAL);
		for (j=0;j<rshape[1];j++) {
			*(double *)PyArray_GETPTR2(output, i, j) = (double)
			NV_Ith_S(y, j);
		}
	}
	N_VDestroy_Serial(y);
	CVodeFree(&cvode_mem);
	free(ic);
	free(par);
	return output;
}


static PyMethodDef cvpyMethods[] = {
	{"run", giccv_run, METH_VARARGS,
		"run the ode. Arguments are ic (tuple of float) -> the initial conditions, pars (tuple of float) -> the parameters, times (1D array of floats) -> the times to report the results at, rna (0 or 1) -> use the rna model or the simple model. If the rna model is used, then ..., otherwise, ic is length 3 (u,v,w) and pars is length 2 (alpha2, epsilon)"},
		{NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initcvpy(void)
{
	import_array();
	(void) Py_InitModule("cvpy", cvpyMethods);
}

