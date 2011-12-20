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
! PAR is alpha2, epsilon, alpha1, n
! U is u, v, w
! F is u', v', w'

       F(1)=PAR(3)/(1 + U(2)**PAR(4)) - U(1)
       F(2)=PAR(3)/(1+ U(3)**PAR(4)) + PAR(1)/(1+U(1)**PAR(4)) - U(2)
       F(3)=PAR(2)*(PAR(3)/(1+U(1)**PAR(4)) - U(3) )   

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
       PAR(2)=.99
       PAR(3)=5.
       PAR(4)=3.

! Initialize the solution
       U(1)=.17442
       U(2)= 3.0245
       U(3)= 4.974

      IF (IEQUIB.NE.0) THEN
        PAR(12) = 1.01555
        PAR(13) = 1.57721
        PAR(14) = 2.44216
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
