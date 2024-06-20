	  implicit real*8(a-h,o-z)
	  parameter(mi=64,mj=64,mk=64)
	  parameter(mkm2=mk-2)
	  parameter(mtot=mi*mj*mk)
	  parameter(Temp=0.05)

	  parameter(nsteps=10000)
	  integer qq(mi,mj,mk) ! chirality
	  integer dope(mi,mj,mk) ! dopant mask
	  integer kount(mk) ! for measuring density of spins in each z-slice
	  integer nconfigs ! 
	  real*8 nx(mi,mj,mk),ny(mi,mj,mk)
	  real*8 nz(mi,mj,mk)
	  real*8 crossx,crossy,crossz,dott
	  real*8 nnx,nny,nnz
	  real*8 d,KK,phitop,rho
	  real*8 rhoz(mk)
	  open(unit=10,file='LLMC-surf01a.dat',status='unknown') ! first parameter output file
	  open(unit=16,file='LLMC-s01a-rhoz-T-0.05.dat',status='unknown') ! plotting density of chiral particles (z)

c     this version has local chirality that is random and switches
c      x=rand()
	  d=0.01d0
	  KK=1.d0
	  pi=4.d0*datan(1.d0)
      rho=0.d0
	  twopi=2.d0*pi

	  ndope=0
c     create random initial conditions with no anchoring at k=1 and k=mk
 	  do 10 i=1,mi
	  do 11 j=1,mj
	  do 12 k=1,mk
	  qq(i,j,k)=1
	  if(rand().le.0.5)then
	    qq(i,j,k)=-1
	  endif	
	  dope(i,j,k)=0
      if(k.eq.1)then
	    dope(i,j,k)=1
		qq(i,j,k)=1
		ndope=ndope+1
	  endif
	  costh=2.d0*(rand()-0.5)
	  sinth=dsqrt(1.d0-costh*costh)
	  phi=rand()*twopi
	  cosphi=dcos(phi)
	  sinphi=dsin(phi)
	  nx(i,j,k)=sinth*cosphi
	  ny(i,j,k)=sinth*sinphi
	  nz(i,j,k)=costh
c     nx(i,j,k)=1.d0
c	  ny(i,j,k)=0.d0
c	  nz(i,j,k)=0.d0
c	  if(k.eq.1)then
c	    nx(i,j,k)=dcos(-phitop/2.d0)
c		ny(i,j,k)=dsin(-phitop/2.d0)
c		nz(i,j,k)=0.d0
c		qq(i,j,k)=1
c		if(mod(i,2).eq.0)qq(i,j,k)=-1
c	  endif
c	  if(k.eq.mk)then
c	    nx(i,j,k)=dcos(phitop/2)
c		ny(i,j,k)=dsin(phitop/2)
c		nz(i,j,k)=0.d0
c		qq(i,j,k)=1
c		if(mod(i,2).ne.0)qq(i,j,k)=-1
c	  endif


12       continue
11      continue
10     continue

       do 15 k=1,mk
	   kount(k)=0
15     continue
       nconfigs=0

c      main loop on MC steps

	   do 999 istep=1,nsteps

	   
	   naccept=0
	   nflip=0
	   
	   do 888 itry=1,mtot
c      pick any site including surface sites	   
	   i=1+floor(rand()*mi)
	   j=1+floor(rand()*mj)
	   k=1+floor(rand()*mk)
	   
	   ip1=mod(i,mi)+1
	   im1=mod(i-2+mi,mi)+1
	   jp1=mod(j,mj)+1
	   jm1=mod(j-2+mj,mj)+1
	   kp1=k+1
	   km1=k-1
c      store current spin
	   nnx=nx(i,j,k)
	   nny=ny(i,j,k)
	   nnz=nz(i,j,k)
	   
	   eold=0.d0

	   dott=nnx*nx(ip1,j,k)+nny*ny(ip1,j,k)+nnz*nz(ip1,j,k)
	   crossx=nny*nz(ip1,j,k)-nnz*ny(ip1,j,k)
	   qqfac=(qq(i,j,k)+qq(ip1,j,k))/2.d0
	   eold=eold+(1.d0-dott*dott)-KK*dott*crossx*qqfac
	   
	   dott=nnx*nx(im1,j,k)+nny*ny(im1,j,k)+nnz*nz(im1,j,k)
	   crossx=nny*nz(im1,j,k)-nnz*ny(im1,j,k)   
	   qqfac=(qq(i,j,k)+qq(im1,j,k))/2.d0
	   eold=eold+(1.d0-dott*dott)+KK*dott*crossx*qqfac
	   
	   dott=nnx*nx(i,jp1,k)+nny*ny(i,jp1,k)+nnz*nz(i,jp1,k)
	   crossy=nnz*nx(i,jp1,k)-nnx*nz(i,jp1,k)
	   qqfac=(qq(i,j,k)+qq(i,jp1,k))/2.d0
	   eold=eold+(1.d0-dott*dott)-KK*dott*crossy*qqfac
	   
	   dott=nnx*nx(i,jm1,k)+nny*ny(i,jm1,k)+nnz*nz(i,jm1,k)
	   crossy=nnz*nx(i,jm1,k)-nnx*nz(i,jm1,k)
	   qqfac=(qq(i,j,k)+qq(i,jm1,k))/2.d0   
	   eold=eold+(1.d0-dott*dott)+KK*dott*crossy*qqfac	   
	   
	   if(kp1.le.mk)then
	   dott=nnx*nx(i,j,kp1)+nny*ny(i,j,kp1)+nnz*nz(i,j,kp1)
	   crossz=nnx*ny(i,j,kp1)-nny*nx(i,j,kp1)
	   qqfac=(qq(i,j,k)+qq(i,j,kp1))/2.d0	   
	   eold=eold+(1.d0-dott*dott)-KK*dott*crossz*qqfac
	   endif
	   
	   if(km1.ge.1)then
	   dott=nnx*nx(i,j,km1)+nny*ny(i,j,km1)+nnz*nz(i,j,km1)
	   crossz=nnx*ny(i,j,km1)-nny*nx(i,j,km1)
	   qqfac=(qq(i,j,k)+qq(i,j,km1))/2.d0	
	   eold=eold+(1.d0-dott*dott)+KK*dott*crossz*qqfac
	   endif
	   
c	   rotate the director at site i,j,k	   

	  if(dabs(nnz).gt.0.999d0)then
	   phi=rand()*twopi
	   xxnew=nnx+d*dcos(phi)
	   yynew=nny+d*dsin(phi)
	   zznew=nnz
	   rsq=dsqrt(xxnew*xxnew+yynew*yynew+zznew*zznew)
	   nnx=xxnew/rsq
	   nny=yynew/rsq
	   nnz=zznew/rsq
	  else
	   ux=-nny
	   uy=nnx
	   uz=0.d0
	   rsq=dsqrt(ux*ux+uy*uy)
	   ux=ux/rsq
	   uy=uy/rsq
	   vx=-nnz*nnx
	   vy=-nnz*nny
	   vz=nnx*nnx+nny*nny
	   rsq=dsqrt(vx*vx+vy*vy+vz*vz)
	   vx=vx/rsq
	   vy=vy/rsq
	   vz=vz/rsq
   
	   phi=rand()*twopi
	   dcosph=d*dcos(phi)
	   dsinph=d*dsin(phi)
	   nnx=nnx+dcosph*ux+dsinph*vx
	   nny=nny+dcosph*uy+dsinph*vy
	   nnz=nnz+dcosph*uz+dsinph*vz
	   rsq=dsqrt(nnx*nnx+nny*nny+nnz*nnz)
	   nnx=nnx/rsq
	   nny=nny/rsq
	   nnz=nnz/rsq	 
c      this is the trial director; could use random matrix rotation instead
       endif	
	   
	   enew=0.d0
	   
	   dott=nnx*nx(ip1,j,k)+nny*ny(ip1,j,k)+nnz*nz(ip1,j,k)
	   crossx=nny*nz(ip1,j,k)-nnz*ny(ip1,j,k)
	   qqfac=(qq(i,j,k)+qq(ip1,j,k))/2.d0
	   enew=enew+(1.d0-dott*dott)-KK*dott*crossx*qqfac
	   
	   dott=nnx*nx(im1,j,k)+nny*ny(im1,j,k)+nnz*nz(im1,j,k)
	   crossx=nny*nz(im1,j,k)-nnz*ny(im1,j,k) 
	   qqfac=(qq(i,j,k)+qq(im1,j,k))/2.d0	   
	   enew=enew+(1.d0-dott*dott)+KK*dott*crossx*qqfac
	   
	   dott=nnx*nx(i,jp1,k)+nny*ny(i,jp1,k)+nnz*nz(i,jp1,k)
	   crossy=nnz*nx(i,jp1,k)-nnx*nz(i,jp1,k)
	   qqfac=(qq(i,j,k)+qq(i,jp1,k))/2.d0
	   enew=enew+(1.d0-dott*dott)-KK*dott*crossy*qqfac
	   
	   dott=nnx*nx(i,jm1,k)+nny*ny(i,jm1,k)+nnz*nz(i,jm1,k)
	   crossy=nnz*nx(i,jm1,k)-nnx*nz(i,jm1,k)
	   qqfac=(qq(i,j,k)+qq(i,jm1,k))/2.d0  
	   enew=enew+(1.d0-dott*dott)+KK*dott*crossy*qqfac	   
	   
	   if(kp1.le.mk)then
	   dott=nnx*nx(i,j,kp1)+nny*ny(i,j,kp1)+nnz*nz(i,j,kp1)
	   crossz=nnx*ny(i,j,kp1)-nny*nx(i,j,kp1)
	   qqfac=(qq(i,j,k)+qq(i,j,kp1))/2.d0	   	   
	   enew=enew+(1.d0-dott*dott)-KK*dott*crossz*qqfac
	   endif
	   if(km1.ge.1)then
	   dott=nnx*nx(i,j,km1)+nny*ny(i,j,km1)+nnz*nz(i,j,km1)
	   crossz=nnx*ny(i,j,km1)-nny*nx(i,j,km1)
	   qqfac=(qq(i,j,k)+qq(i,j,km1))/2.d0	
	   enew=enew+(1.d0-dott*dott)+KK*dott*crossz*qqfac  
	   endif	   
	   	   
	   
	   if(enew.lt.eold)then
c        accept the change
	     nx(i,j,k)=nnx
		 ny(i,j,k)=nny
		 nz(i,j,k)=nnz
		 naccept=naccept+1

	   else
	     delta=enew-eold
		 pp=dexp(-delta/Temp)
		 if(rand().le.pp)then
	      nx(i,j,k)=nnx
		  ny(i,j,k)=nny
		  nz(i,j,k)=nnz
		  naccept=naccept+1
		 endif
	   endif
	   
c      pick a new spin and try switching its chirality
	   
	   i=1+floor(rand()*mi)
	   j=1+floor(rand()*mj)
	   k=1+floor(rand()*mk)
	   if(dope(i,j,k).eq.0)then
	   ip1=mod(i,mi)+1
	   im1=mod(i-2+mi,mi)+1
	   jp1=mod(j,mj)+1
	   jm1=mod(j-2+mj,mj)+1
	   kp1=k+1
	   km1=k-1	   
	   
c      store current spin
	   nnx=nx(i,j,k)
	   nny=ny(i,j,k)
	   nnz=nz(i,j,k)
	   
	   eold=0.d0

	   dott=nnx*nx(ip1,j,k)+nny*ny(ip1,j,k)+nnz*nz(ip1,j,k)
	   crossx=nny*nz(ip1,j,k)-nnz*ny(ip1,j,k)
	   qqfac=(qq(i,j,k)+qq(ip1,j,k))/2.d0
	   eold=eold+(1.d0-dott*dott)-KK*dott*crossx*qqfac

	   
	   dott=nnx*nx(im1,j,k)+nny*ny(im1,j,k)+nnz*nz(im1,j,k)
	   crossx=nny*nz(im1,j,k)-nnz*ny(im1,j,k)   
	   qqfac=(qq(i,j,k)+qq(im1,j,k))/2.d0
	   eold=eold+(1.d0-dott*dott)+KK*dott*crossx*qqfac

	   
	   dott=nnx*nx(i,jp1,k)+nny*ny(i,jp1,k)+nnz*nz(i,jp1,k)
	   crossy=nnz*nx(i,jp1,k)-nnx*nz(i,jp1,k)
	   qqfac=(qq(i,j,k)+qq(i,jp1,k))/2.d0
	   eold=eold+(1.d0-dott*dott)-KK*dott*crossy*qqfac

	   
	   dott=nnx*nx(i,jm1,k)+nny*ny(i,jm1,k)+nnz*nz(i,jm1,k)
	   crossy=nnz*nx(i,jm1,k)-nnx*nz(i,jm1,k)
	   qqfac=(qq(i,j,k)+qq(i,jm1,k))/2.d0   
	   eold=eold+(1.d0-dott*dott)+KK*dott*crossy*qqfac	 
	 

       if(kp1.le.mk)then	   
	   dott=nnx*nx(i,j,kp1)+nny*ny(i,j,kp1)+nnz*nz(i,j,kp1)
	   crossz=nnx*ny(i,j,kp1)-nny*nx(i,j,kp1)
	   qqfac=(qq(i,j,k)+qq(i,j,kp1))/2.d0	   
	   eold=eold+(1.d0-dott*dott)-KK*dott*crossz*qqfac

	   endif
	   
	   if(km1.ge.1)then
	   dott=nnx*nx(i,j,km1)+nny*ny(i,j,km1)+nnz*nz(i,j,km1)
	   crossz=nnx*ny(i,j,km1)-nny*nx(i,j,km1)
	   qqfac=(qq(i,j,k)+qq(i,j,km1))/2.d0	
	   eold=eold+(1.d0-dott*dott)+KK*dott*crossz*qqfac

	   endif
	   
c	   switch the chirality

       qqnew=-qq(i,j,k)
	   
	   enew=0.d0
	   
	   dott=nnx*nx(ip1,j,k)+nny*ny(ip1,j,k)+nnz*nz(ip1,j,k)
	   crossx=nny*nz(ip1,j,k)-nnz*ny(ip1,j,k)
	   qqfac=(qqnew+qq(ip1,j,k))/2.d0
	   enew=enew+(1.d0-dott*dott)-KK*dott*crossx*qqfac
)
	   
	   dott=nnx*nx(im1,j,k)+nny*ny(im1,j,k)+nnz*nz(im1,j,k)
	   crossx=nny*nz(im1,j,k)-nnz*ny(im1,j,k) 
	   qqfac=(qqnew+qq(im1,j,k))/2.d0	   
	   enew=enew+(1.d0-dott*dott)+KK*dott*crossx*qqfac

	   
	   dott=nnx*nx(i,jp1,k)+nny*ny(i,jp1,k)+nnz*nz(i,jp1,k)
	   crossy=nnz*nx(i,jp1,k)-nnx*nz(i,jp1,k)
	   qqfac=(qqnew+qq(i,jp1,k))/2.d0
	   enew=enew+(1.d0-dott*dott)-KK*dott*crossy*qqfac

	   
	   dott=nnx*nx(i,jm1,k)+nny*ny(i,jm1,k)+nnz*nz(i,jm1,k)
	   crossy=nnz*nx(i,jm1,k)-nnx*nz(i,jm1,k)
	   qqfac=(qqnew+qq(i,jm1,k))/2.d0  
	   enew=enew+(1.d0-dott*dott)+KK*dott*crossy*qqfac
   
	   
	   if(kp1.le.mk)then
	   dott=nnx*nx(i,j,kp1)+nny*ny(i,j,kp1)+nnz*nz(i,j,kp1)
	   crossz=nnx*ny(i,j,kp1)-nny*nx(i,j,kp1)
	   qqfac=(qqnew+qq(i,j,kp1))/2.d0	   	   
	   enew=enew+(1.d0-dott*dott)-KK*dott*crossz*qqfac

	   endif
	   
	   if(km1.ge.1)then
	   dott=nnx*nx(i,j,km1)+nny*ny(i,j,km1)+nnz*nz(i,j,km1)
	   crossz=nnx*ny(i,j,km1)-nny*nx(i,j,km1)
	   qqfac=(qqnew+qq(i,j,km1))/2.d0	
	   enew=enew+(1.d0-dott*dott)+KK*dott*crossz*qqfac 
	   
	   endif	   
	   
	   if(enew.lt.eold)then
c        accept the change
       qq(i,j,k)=qqnew
	   nflip=nflip+1
	   else
	     delta=enew-eold
		 pp=dexp(-delta/Temp)
		 if(rand().le.pp)then
          qq(i,j,k)=qqnew
          nflip=nflip+1
		 endif
	   endif
	   	   
	   
	   endif
		 
888    continue

c      end of MC step

	   paccept=float(naccept)/float(mtot)
	   if(paccept.lt.0.4)then
	     d=d*0.995
	   elseif(paccept.gt.0.6)then
	     d=d/0.995
	   endif

	   if(mod(istep,50).eq.0)then
	   
c      calculate total energy
       etot=0.d0

	   do 100 i=1,mi
	   do 101 j=1,mj
	   do 102 k=1,mk

	   ip1=mod(i,mi)+1
	   jp1=mod(j,mj)+1
	   kp1=k+1

c      store current spin
	   nnx=nx(i,j,k)
	   nny=ny(i,j,k)
	   nnz=nz(i,j,k)
	   
	   dott=nnx*nx(ip1,j,k)+nny*ny(ip1,j,k)+nnz*nz(ip1,j,k)
	   crossx=nny*nz(ip1,j,k)-nnz*ny(ip1,j,k)
	   qqfac=(qq(i,j,k)+qq(ip1,j,k))/2.d0
	   etot=etot+(1.d0-dott*dott)-KK*dott*crossx*qqfac

	   
	   dott=nnx*nx(i,jp1,k)+nny*ny(i,jp1,k)+nnz*nz(i,jp1,k)
	   crossy=nnz*nx(i,jp1,k)-nnx*nz(i,jp1,k)
	   qqfac=(qq(i,j,k)+qq(i,jp1,k))/2.d0
	   etot=etot+(1.d0-dott*dott)-KK*dott*crossy*qqfac

       if(kp1.le.mk)then
	   dott=nnx*nx(i,j,kp1)+nny*ny(i,j,kp1)+nnz*nz(i,j,kp1)
	   crossz=nnx*ny(i,j,kp1)-nny*nx(i,j,kp1)
	   qqfac=(qq(i,j,k)+qq(i,j,kp1))/2.d0	   
	   etot=etot+(1.d0-dott*dott)-KK*dott*crossz*qqfac

	   endif
102    continue
101    continue
100    continue

       faccept=float(nflip)/float(mtot)

	   etot=etot/float(mtot)
	   
c      calculate enantiomeric excess

       e_excess=0.d0
       do 300 i=1,mi
	   do 301 j=1,mj
	   do 302 k=2,mk-1
	   e_excess=e_excess+qq(i,j,k)
302    continue
301    continue
300    continue
	   e_excess=e_excess/float(mi*mj*(mk-2))
	   
	   write(6,2000)istep,etot,e_excess,paccept,faccept,d	
       write(10,*)istep,etot,e_excess	   
2000   format(1x,i6,1x,f8.4,1x,f8.4,1x,f8.4,1x,f8.4,1x,f8.4)

       if(istep.gt.5000)then
	    nconfigs=nconfigs+1
		do 500 i=1,mi
		 do 501 j=1,mj
		  do 502 k=1,mk
		  if(qq(i,j,k).eq.1)kount(k)=kount(k)+1
502		  continue
501		 continue
500		continue
	   endif

       endif
	   

	   
999    continue

       do 550 k=1,mk
	   rhoz(k)=float(kount(k))/float(mi*mj*nconfigs)
       write(6,*)k,kount(k),rhoz(k)
	   write(16,3000)k,rhoz(k)
3000   format(1x,i6,1x,f8.4)
550    continue

c      end of isteps. Print config and close.

	  open(unit=11,file='LLMCsurface-configa.dat',status='unknown')
	  open(unit=12,file='LLMCsurface-configb.dat',status='unknown')

       scale=0.4
	   do 200 i=1,mi
	   j=mj/2
	   do 201 k=1,mk
	   
	   x1=i-scale*nx(i,j,k)
	   x2=i+scale*nx(i,j,k)
	   z1=k-scale*nz(i,j,k)
	   z2=k+scale*nz(i,j,k)

       if(qq(i,j,k).eq.1)then
	   write(11,*)x1,z1
	   write(11,*)x2,z2
	   write(11,*)
	   else
	   write(12,*)x1,z1
	   write(12,*)x2,z2
	   write(12,*)
       endif	   
	   
	   
201    continue
200    continue

	   close(unit=10)
	   close(unit=11)
	   close(unit=12)
	   close(unit=16)

	   stop
	   end
	   
	   

	  
	  
	
