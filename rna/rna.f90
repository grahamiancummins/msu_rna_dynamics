!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   ab :            The A --> B reaction 
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

! Evaluates the algebraic equations or ODE right hand side

! Input arguments :
!      NDIM   :   Dimension of the ODE system 
!      U      :   State variables
!      ICP    :   Array indicating the free parameter(s)
!      PAR    :   Equation parameters

! Values to be returned :
!      F      :   ODE right hand side values

! Normally unused Jacobian arguments : IJAC, DFDU, DFDP (see manual)

      IMPLICIT NONE
      INTEGER NDIM, IJAC, ICP(*)
      DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)
      DOUBLE PRECISION N, AL1, A

! PAR is alpha2, epsilon, r, r_m
! fixed values a =1, alpha1=5, n = 3
! U is u_rna, u, v_rna, v,  w
! F is u_rna', u', v_rna', v', w'
       A=1.0
       AL1=5.0
       N=3.0
       F(1)=PAR(4)*(AL1/(1 + U(4)**N) - U(1))
       F(2)=PAR(3)*(A*U(1)-U(2))
       F(3)=PAR(4)*(AL1/(1+ U(5)**N) + PAR(1)/(1+U(2)**N) - U(3))
       F(4)=PAR(3)*(A*U(3)-U(4))
       F(5)=PAR(2)*(AL1/(1+U(2)**N) - U(5) )   
      END SUBROUTINE FUNC
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

! Input arguments :
!      NDIM   :   Dimension of the ODE system 

! Values to be returned :
!      U      :   A starting solution vector
!      PAR    :   The corresponding equation-parameter values
!      T      :   Not used here

      IMPLICIT NONE
      INTEGER NDIM
      DOUBLE PRECISION U(NDIM), PAR(*), T
      INTEGER ITWIST,ISTART,IEQUIB,NFIXED,NPSI,NUNSTAB,NSTAB,NREV
      COMMON /BLHOM/ ITWIST,ISTART,IEQUIB,NFIXED,NPSI,NUNSTAB,NSTAB,NREV
! Initialize the equation parameters
       PAR(1)=3.
       PAR(2)=.05
       PAR(3)=1.0
       PAR(4)=1.0

! Initialize the solution
       U(1)=.17442
       U(2)=.17442
       U(3)= 3.0245
       U(4)= 3.0245
       U(5)= 4.974

      IF (IEQUIB.NE.0) THEN
! This is at e=.99, r=1. Not sure if that's good for any r
        PAR(12) = 1.12635
        PAR(13) = 1.12635
        PAR(14) = 1.50944
        PAR(15) = 1.50944
        PAR(16) = 2.05848
      ENDIF
      END SUBROUTINE STPNT
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! The following subroutines are not used here,
! but they must be supplied as dummy routines

      SUBROUTINE BCND 
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
!----------------------------------------------------------------------
!----------------------------------------------------------------------
