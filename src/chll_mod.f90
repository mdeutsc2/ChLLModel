module types
	implicit none
	save

	integer,parameter :: int32 = selected_int_kind(8)
	integer,parameter :: int64 = selected_int_kind(15)
	integer,parameter :: real64 = selected_real_kind(15)
end module types

module chll_mod
	use types
	
	contains
	subroutine init(nx,ny,nz,s,dope,sl1,sl2,ni,nj,nk)
	  implicit none
	  real(kind=real64), intent(in out) :: nx(:,:,:),ny(:,:,:),nz(:,:,:)
	  integer(kind=int64), intent(in out) :: s(:,:,:),dope(:,:,:),sl1(:),sl2(:)
	  integer(kind=int64), intent(in) :: ni,nj,nk
	  real(kind=real64) :: rnd,costh,sinth,phi,cosphi,sinphi,pi,twopi
	  integer(kind=int64) :: i,j,k,nsub1,nsub2,idx
	  pi = 4.0d0*datan(1.0d0)
	  twopi = 2.0d0*pi
	  ! setting up initial random state
	  do i = 1,ni
		 do j = 1,nj
			do k = 1,nk
			   s(i,j,k) = 1
			   call random_number(rnd)
			   if (rnd.le.0.5d0) then
				  s(i,j,k) = -1
			   endif
			   dope(i,j,k) = 0
			   call random_number(rnd)
			   costh = 2.d0*(rnd-0.5)
			   sinth = dsqrt(1.d0-costh*costh)
			   call random_number(rnd)
			   phi = rnd*twopi
			   cosphi = dcos(phi)
			   sinphi = dsin(phi)
			   nx(i,j,k) = sinth*cosphi
			   ny(i,j,k) = sinth*sinphi
			   nz(i,j,k) = costh
			enddo
		 enddo
	  enddo

	  ! create list of site indices for each sublattice, sub1, sub2
	  nsub1 = 0
	  nsub2 = 0
	  do k = 1,nk
		 do j = 1,nj
			do i = 1,ni
			   idx = i+(j-1)*ni + (k-1)*(ni*nj)
			   if (mod(i+j+k,2).ne.0) then
				  nsub1 = nsub1+1
				  sl1(nsub1) = idx
			   else
				  nsub2 = nsub2 + 1
				  sl2(nsub2) = idx
			   endif
			enddo
		 enddo
	  enddo
	  return
	end subroutine init


	subroutine run(nsteps,nx,ny,nz,s,dope,sl1,sl2,rand1,rand2,KK,d,kbt,ni,nj,nk)
		implicit none
		integer(kind=int64), intent(in) :: nsteps,ni,nj,nk
		real(kind=real64), intent(in out) :: nx(:,:,:),ny(:,:,:),nz(:,:,:),rand1(:,:),rand2(:,:)
		real(kind=real64), intent(in out) :: KK,d,kbt
		integer(kind=int64), intent(in out) :: s(:,:,:),dope(:,:,:),sl1(:),sl2(:)
		integer(kind=int64) :: naccept,nflip,istep,nsub,n3
		real(kind=real64) :: paccept,faccept,total_energy,e_excess
		n3 = ni*nj*nk
		nsub = n3/2
		!$acc enter data copyin(nx,ny,nz,s,sl1,sl2,rand1,rand2)
		do istep = 1,nsteps
		 naccept = 0!(:,:,:) = 0
		 nflip = 0!(:,:,:) = 0
		 !do itry = 1,n3
		 call random_number(rand1)
		 call random_number(rand2)
		 !$acc update device(rand1,rand2)
		 call evolve(sl1,nx,ny,nz,s,rand1,rand2,KK,d,kbt,nsub,naccept,nflip,ni,nj,nk)
		 !$acc update device(rand1,rand2)
		 call random_number(rand1)
		 call random_number(rand2)
		 call evolve(sl2,nx,ny,nz,s,rand1,rand2,KK,d,kbt,nsub,naccept,nflip,ni,nj,nk)
		 !paccept = float(sum(naccept))/float(n3) ! % of accepted director rotation
		 paccept = float(naccept)/float(n3)
		 if (paccept.lt.0.4d0) then
			d = d*0.995
		 elseif (paccept.gt.0.6d0) then
			d = d/0.995
		 endif
		 if (mod(istep,100).eq.0) then
			!faccept = float(sum(nflip))/float(n3)
			faccept = float(nflip)/float(n3)
			! calculate total energy
			total_energy = etot(nx,ny,nz,s,KK,ni,nj,nk)
			e_excess = sum(float(s))/float(ni*nj*nk)
			print*,istep,total_energy,e_excess,paccept,faccept,d
			!print*,istep,total_energy,e_excess,sum(naccept),sum(nflip)
		 ! calculate enantiomeric excess
		 endif
		enddo
		!$acc exit data copyout(nx,ny,nz,s,sl1,sl2,rand1,rand2)
	end subroutine run

	subroutine evolve(sl,nx,ny,nz,s,rand1,rand2,KK,d,kbt,nsub,naccept,nflip,ni,nj,nk)
		implicit none
		integer(kind=int64), intent(in) :: sl(:),nsub
		real(kind=real64),intent(in out) :: nx(:,:,:),ny(:,:,:),nz(:,:,:)
		integer(kind=int64), intent(in out) :: s(:,:,:),naccept,nflip
		real(kind=real64),intent(in) :: KK,d,rand1(:,:),rand2(:,:),kbt
		integer(kind=int64),intent(in) :: ni,nj,nk
		real(kind=real64) :: dcosphi,dsinphi,enew,eold,nnx,nny,nnz,ux,uy,uz,vx,vy,vz,xxnew,yynew,zznew,rsq,phi
		integer(kind=int64) :: itry,i,j,k,ip1,im1,jp1,jm1,kp1,km1,snew,index
		real(kind=real64) :: dott,crossx,crossy,crossz,sfac,pi,twopi
	    pi = 4.0d0*datan(1.0d0)
	    twopi = 2.0d0*pi
		!do itry = 1,nsub
		do concurrent(itry = 1:nsub) local(ip1,im1,jp1,jm1,kp1,km1,nnx,nny,nnz,dott,crossx,crossy,crossz,eold,enew,sfac) shared(nx,ny,nz,s,sl,rand1,rand2) reduce(+:naccept) reduce(+:nflip)
			index = sl(itry)
			i = 1+mod(index-1,ni)
			j = 1+int(mod((index-1),(ni*nj))/(ni))
			k = 1+int((index-1)/(ni*nj))
			!call evolve_director(i,j,k,nx,ny,nz,s,naccept,kbt,KK,d)
			ip1 = i+1 !mod(i,ni) + 1
			im1 = i-1 !mod(i-2+ni,ni)+1
			jp1 = j+1 !mod(j,nj)+1
			jm1 = j-1 !mod(j-2+ni,ni)+1
			kp1 = k + 1
			km1 = k - 1

			! store the current spin and set energy to zero
			nnx = nx(i,j,k)
			nny = ny(i,j,k)
			nnz = nz(i,j,k)
			#eold = energy(nx,ny,nz,nnx,nny,nnz,s(i,j,k),s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
			eold = 0.0d0
			if (ip1 .le. ni) then
				dott = nnx*nx(ip1,j,k) + nny*ny(ip1,j,k) + nnz*nz(ip1,j,k)
				crossx = nny*nz(ip1,j,k) - nnz*ny(ip1,j,k)
				sfac = 0.5d0*(s(i,j,k)+s(ip1,j,k))
				eold = eold + (1.0d0-dott*dott)-KK*dott*crossx*sfac
			endif
			! im1
			if (im1 .ge. 1) then
				   dott = nnx*nx(im1,j,k) + nny*ny(im1,j,k) + nnz*nz(im1,j,k)
				   crossx = nny*nz(im1,j,k) - nnz*ny(im1,j,k)
				   sfac = 0.5d0*(s(i,j,k)+s(im1,j,k))
				   eold = eold + (1.0d0-dott*dott)+KK*dott*crossx*sfac
			endif
			! jp1
			if (jp1 .le. nj) then
				   dott = nnx*nx(i,jp1,k) + nny*ny(i,jp1,k) + nnz*nz(i,jp1,k)
				   crossy = nnz*nx(i,jp1,k) - nnx*nz(i,jp1,k)
				   sfac = 0.5d0*(s(i,j,k)+s(i,jp1,k))
				   eold = eold + (1.0d0-dott*dott)-KK*dott*crossy*sfac
			endif
			! jm1
			if (jm1 .ge. 1) then
				   dott = nnx*nx(i,jm1,k) + nny*ny(i,jm1,k) + nnz*nz(i,jm1,k)
				   crossy = nnz*nx(i,jm1,k) - nnx*nz(i,jm1,k)
				   sfac = 0.5d0*(s(i,j,k) + s(i,jm1,k))
				   eold = eold + (1.0d0-dott*dott)+KK*dott*crossy*sfac
			endif
			!kp1	
			if (kp1.le.nk) then
				   dott = nnx*nx(i,j,kp1) + nny*ny(i,j,kp1) + nnz*nz(i,j,kp1)
				   crossz = nnx*ny(i,j,kp1) - nny*nx(i,j,kp1)
				   sfac = 0.5d0*(s(i,j,k)+s(i,j,kp1))
				   eold = eold + (1.0d0-dott*dott)-KK*dott*crossz*sfac
			endif
			!km1
			if (km1.ge.1) then
				   dott = nnx*nx(i,j,km1) + nny*ny(i,j,km1) + nnz*nz(i,j,km1)
				   crossz = nnx*ny(i,j,km1) - nny*nx(i,j,km1)
				   sfac = 0.5d0*(s(i,j,k)+s(i,j,km1))
				   eold = eold + (1.0d0-dott*dott)+KK*dott*crossz*sfac
			endif
			! rotate the director at site i,j,k to get trial director (stored in nnx,nny,nnz)
			if (abs(nnz).gt.0.999d0) then
			   phi = rand1(itry,1)*twopi
			   xxnew = nnx+d*dcos(phi)
			   yynew = nny+d*dsin(phi)
			   zznew = nnz
			   rsq = dsqrt(xxnew*xxnew+yynew*yynew+zznew*zznew)
			   nnx = xxnew/rsq
			   nny = yynew/rsq
			   nnz = zznew/rsq
			else
			   ux = -nny
			   uy = nnx
			   uz = 0.0d0
			   rsq = dsqrt(ux*ux+uy*uy)
			   ux = ux/rsq
			   uy = uy/rsq
			   uz = uz/rsq
			   vx = -nnz*nnx
			   vy = -nnz*nny
			   vz = nnx*nnx+nny*nny
			   rsq = dsqrt(vx*vx+vy*vy+vz*vz)
			   vx = vx/rsq
			   vy = vy/rsq
			   vz = vz/rsq
			   phi = rand1(itry,2)*twopi
			   dcosphi = d*dcos(phi)
			   dsinphi = d*dsin(phi)
			   nnx = nnx+dcosphi*ux + dsinphi*vx
			   nny = nny+dcosphi*uy + dsinphi*vy
			   nnz = nnz+dcosphi*uz + dsinphi*vz
			   rsq = dsqrt(nnx*nnx+nny*nny+nnz*nnz)
			   nnx = nnx/rsq
			   nny = nny/rsq
			   nnz = nnz/rsq
			endif
			! calculate enew w/ trial spin
			!enew = energy(nx,ny,nz,nnx,nny,nnz,s(i,j,k),s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
			enew = 0.0d0
			if (ip1 .le. ni) then
				dott = nnx*nx(ip1,j,k) + nny*ny(ip1,j,k) + nnz*nz(ip1,j,k)
				crossx = nny*nz(ip1,j,k) - nnz*ny(ip1,j,k)
				sfac = 0.5d0*(s(i,j,k)+s(ip1,j,k))
				enew = enew + (1.0d0-dott*dott)-KK*dott*crossx*sfac
			endif
			! im1
			if (im1 .ge. 1) then
				   dott = nnx*nx(im1,j,k) + nny*ny(im1,j,k) + nnz*nz(im1,j,k)
				   crossx = nny*nz(im1,j,k) - nnz*ny(im1,j,k)
				   sfac = 0.5d0*(s(i,j,k)+s(im1,j,k))
				   enew = enew + (1.0d0-dott*dott)+KK*dott*crossx*sfac
			endif
			! jp1
			if (jp1 .le. nj) then
				   dott = nnx*nx(i,jp1,k) + nny*ny(i,jp1,k) + nnz*nz(i,jp1,k)
				   crossy = nnz*nx(i,jp1,k) - nnx*nz(i,jp1,k)
				   sfac = 0.5d0*(s(i,j,k)+s(i,jp1,k))
				   enew = enew + (1.0d0-dott*dott)-KK*dott*crossy*sfac
			endif
			! jm1
			if (jm1 .ge. 1) then
				   dott = nnx*nx(i,jm1,k) + nny*ny(i,jm1,k) + nnz*nz(i,jm1,k)
				   crossy = nnz*nx(i,jm1,k) - nnx*nz(i,jm1,k)
				   sfac = 0.5d0*(s(i,j,k) + s(i,jm1,k))
				   enew = enew + (1.0d0-dott*dott)+KK*dott*crossy*sfac
			endif
			!kp1	
			if (kp1.le.nk) then
				   dott = nnx*nx(i,j,kp1) + nny*ny(i,j,kp1) + nnz*nz(i,j,kp1)
				   crossz = nnx*ny(i,j,kp1) - nny*nx(i,j,kp1)
				   sfac = 0.5d0*(s(i,j,k)+s(i,j,kp1))
				   enew = enew + (1.0d0-dott*dott)-KK*dott*crossz*sfac
			endif
			!km1
			if (km1.ge.1) then
				   dott = nnx*nx(i,j,km1) + nny*ny(i,j,km1) + nnz*nz(i,j,km1)
				   crossz = nnx*ny(i,j,km1) - nny*nx(i,j,km1)
				   sfac = 0.5d0*(s(i,j,k)+s(i,j,km1))
				   enew = enew + (1.0d0-dott*dott)+KK*dott*crossz*sfac
			endif
		!print*,i,j,k,eold,enew,enew.lt.eold
		! metropolis algorithm
		if (enew.lt.eold) then
		   nx(i,j,k) = nnx
		   ny(i,j,k) = nny
		   nz(i,j,k) = nnz
		   naccept = naccept + 1
		 else
			if (rand2(itry,1).le.dexp(-(enew-eold)/kbt)) then
			  nx(i,j,k) = nnx
			  ny(i,j,k) = nny
			  nz(i,j,k) = nnz
			  naccept = naccept + 1
			endif
		 endif
		 
		 !call evolve_chirality(i,j,k,nx,ny,nz,s,nflip,kbt,KK)
					! monte-carlo for switching chirality
				   !ip1 = i+1 !mod(i,ni) + 1
				   !im1 = i-1 !mod(i-2+ni,ni)+1
				   !jp1 = j+1 !mod(j,nj)+1
				   !jm1 = j-1 !mod(j-2+ni,ni)+1
				   !kp1 = k + 1
				   !km1 = k - 1
			! calculate eold
			nnx = nx(i,j,k)
			nny = ny(i,j,k)
			nnz = nz(i,j,k)
			!eold = energy(nx,ny,nz,nnx,nny,nnz,s(i,j,k),s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
			eold = 0.0d0
			if (ip1 .le. ni) then
				dott = nnx*nx(ip1,j,k) + nny*ny(ip1,j,k) + nnz*nz(ip1,j,k)
				crossx = nny*nz(ip1,j,k) - nnz*ny(ip1,j,k)
				sfac = 0.5d0*(s(i,j,k)+s(ip1,j,k))
				eold = eold + (1.0d0-dott*dott)-KK*dott*crossx*sfac
			endif
			! im1
			if (im1 .ge. 1) then
				   dott = nnx*nx(im1,j,k) + nny*ny(im1,j,k) + nnz*nz(im1,j,k)
				   crossx = nny*nz(im1,j,k) - nnz*ny(im1,j,k)
				   sfac = 0.5d0*(s(i,j,k)+s(im1,j,k))
				   eold = eold + (1.0d0-dott*dott)+KK*dott*crossx*sfac
			endif
			! jp1
			if (jp1 .le. nj) then
				   dott = nnx*nx(i,jp1,k) + nny*ny(i,jp1,k) + nnz*nz(i,jp1,k)
				   crossy = nnz*nx(i,jp1,k) - nnx*nz(i,jp1,k)
				   sfac = 0.5d0*(s(i,j,k)+s(i,jp1,k))
				   eold = eold + (1.0d0-dott*dott)-KK*dott*crossy*sfac
			endif
			! jm1
			if (jm1 .ge. 1) then
				   dott = nnx*nx(i,jm1,k) + nny*ny(i,jm1,k) + nnz*nz(i,jm1,k)
				   crossy = nnz*nx(i,jm1,k) - nnx*nz(i,jm1,k)
				   sfac = 0.5d0*(s(i,j,k) + s(i,jm1,k))
				   eold = eold + (1.0d0-dott*dott)+KK*dott*crossy*sfac
			endif
			!kp1	
			if (kp1.le.nk) then
				   dott = nnx*nx(i,j,kp1) + nny*ny(i,j,kp1) + nnz*nz(i,j,kp1)
				   crossz = nnx*ny(i,j,kp1) - nny*nx(i,j,kp1)
				   sfac = 0.5d0*(s(i,j,k)+s(i,j,kp1))
				   eold = eold + (1.0d0-dott*dott)-KK*dott*crossz*sfac
			endif
			!km1
			if (km1.ge.1) then
				   dott = nnx*nx(i,j,km1) + nny*ny(i,j,km1) + nnz*nz(i,j,km1)
				   crossz = nnx*ny(i,j,km1) - nny*nx(i,j,km1)
				   sfac = 0.5d0*(s(i,j,k)+s(i,j,km1))
				   eold = eold + (1.0d0-dott*dott)+KK*dott*crossz*sfac
			endif
			! switch chirality and calculate enew
			snew = -s(i,j,k)
			!enew = energy(nx,ny,nz,nnx,nny,nnz,snew,s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
			enew = 0.0d0
			if (ip1 .le. ni) then
				dott = nnx*nx(ip1,j,k) + nny*ny(ip1,j,k) + nnz*nz(ip1,j,k)
				crossx = nny*nz(ip1,j,k) - nnz*ny(ip1,j,k)
				sfac = 0.5d0*(snew+s(ip1,j,k))
				enew = enew + (1.0d0-dott*dott)-KK*dott*crossx*sfac
			endif
			! im1
			if (im1 .ge. 1) then
				   dott = nnx*nx(im1,j,k) + nny*ny(im1,j,k) + nnz*nz(im1,j,k)
				   crossx = nny*nz(im1,j,k) - nnz*ny(im1,j,k)
				   sfac = 0.5d0*(snew+s(im1,j,k))
				   enew = enew + (1.0d0-dott*dott)+KK*dott*crossx*sfac
			endif
			! jp1
			if (jp1 .le. nj) then
				   dott = nnx*nx(i,jp1,k) + nny*ny(i,jp1,k) + nnz*nz(i,jp1,k)
				   crossy = nnz*nx(i,jp1,k) - nnx*nz(i,jp1,k)
				   sfac = 0.5d0*(snew+s(i,jp1,k))
				   enew = enew + (1.0d0-dott*dott)-KK*dott*crossy*sfac
			endif
			! jm1
			if (jm1 .ge. 1) then
				   dott = nnx*nx(i,jm1,k) + nny*ny(i,jm1,k) + nnz*nz(i,jm1,k)
				   crossy = nnz*nx(i,jm1,k) - nnx*nz(i,jm1,k)
				   sfac = 0.5d0*(snew + s(i,jm1,k))
				   enew = enew + (1.0d0-dott*dott)+KK*dott*crossy*sfac
			endif
			!kp1	
			if (kp1.le.nk) then
				   dott = nnx*nx(i,j,kp1) + nny*ny(i,j,kp1) + nnz*nz(i,j,kp1)
				   crossz = nnx*ny(i,j,kp1) - nny*nx(i,j,kp1)
				   sfac = 0.5d0*(snew+s(i,j,kp1))
				   enew = enew + (1.0d0-dott*dott)-KK*dott*crossz*sfac
			endif
			!km1
			if (km1.ge.1) then
				   dott = nnx*nx(i,j,km1) + nny*ny(i,j,km1) + nnz*nz(i,j,km1)
				   crossz = nnx*ny(i,j,km1) - nny*nx(i,j,km1)
				   sfac = 0.5d0*(snew+s(i,j,km1))
				   enew = enew + (1.0d0-dott*dott)+KK*dott*crossz*sfac
			endif
			! metropolis
			if (enew.lt.eold) then
			   s(i,j,k) = snew
			   nflip = nflip + 1
			else
			   if (rand2(itry,2).le.dexp(-(enew-eold)/kbt)) then
				  s(i,j,k) = snew
				  nflip = nflip + 1
			   endif
			endif
		 enddo ! end 1st sublattice
	end subroutine evolve

	pure real function energy(nx,ny,nz,nnx,nny,nnz,snew,sold,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
    implicit none
    integer(kind=int64),intent(in) :: i,j,k,ip1,im1,jp1,jm1,kp1,km1,snew,sold(:,:,:),ni,nj,nk
    real(kind=real64),intent(in) :: nx(:,:,:),ny(:,:,:),nz(:,:,:),nnx,nny,nnz,KK
    real(kind=real64) :: dott,crossx,crossy,crossz,sfac,etmp
    etmp = 0.0d0
    ! ip1
    if (ip1 .le. ni) then
       dott = nnx*nx(ip1,j,k) + nny*ny(ip1,j,k) + nnz*nz(ip1,j,k)
       crossx = nny*nz(ip1,j,k) - nnz*ny(ip1,j,k)
       sfac = 0.5d0*(snew+sold(ip1,j,k))
       etmp = etmp + (1.0d0-dott*dott)-KK*dott*crossx*sfac
    endif
    ! im1
    if (im1 .ge. 1) then
       dott = nnx*nx(im1,j,k) + nny*ny(im1,j,k) + nnz*nz(im1,j,k)
       crossx = nny*nz(im1,j,k) - nnz*ny(im1,j,k)
       sfac = 0.5d0*(snew+sold(im1,j,k))
       etmp = etmp + (1.0d0-dott*dott)+KK*dott*crossx*sfac
    endif
    ! jp1
    if (jp1 .le. nj) then
       dott = nnx*nx(i,jp1,k) + nny*ny(i,jp1,k) + nnz*nz(i,jp1,k)
       crossy = nnz*nx(i,jp1,k) - nnx*nz(i,jp1,k)
       sfac = 0.5d0*(snew+sold(i,jp1,k))
       etmp = etmp + (1.0d0-dott*dott)-KK*dott*crossy*sfac
    endif
    ! jm1
    if (jm1 .ge. 1) then
       dott = nnx*nx(i,jm1,k) + nny*ny(i,jm1,k) + nnz*nz(i,jm1,k)
       crossy = nnz*nx(i,jm1,k) - nnx*nz(i,jm1,k)
       sfac = 0.5d0*(snew+sold(i,jm1,k))
       etmp = etmp + (1.0d0-dott*dott)+KK*dott*crossy*sfac
    endif
    
    if (kp1.le.nk) then
       dott = nnx*nx(i,j,kp1) + nny*ny(i,j,kp1) + nnz*nz(i,j,kp1)
       crossz = nnx*ny(i,j,kp1) - nny*nx(i,j,kp1)
       sfac = 0.5d0*(snew+sold(i,j,kp1))
       etmp = etmp + (1.0d0-dott*dott)-KK*dott*crossz*sfac
    endif
    if (km1.ge.1) then
       dott = nnx*nx(i,j,km1) + nny*ny(i,j,km1) + nnz*nz(i,j,km1)
       crossz = nnx*ny(i,j,km1) - nny*nx(i,j,km1)
       sfac = 0.5d0*(snew+sold(i,j,km1))
       etmp = etmp + (1.0d0-dott*dott)+KK*dott*crossz*sfac
    endif
    energy = etmp
  end function energy

  real function etot(nx,ny,nz,s,KK,ni,nj,nk)
    implicit none
    real(kind=real64),intent(in) :: nx(:,:,:),ny(:,:,:),nz(:,:,:),KK
    integer(kind=int64),intent(in) :: s(:,:,:),ni,nj,nk
    real(kind=real64) :: ddot,crossx,crossy,crossz,sfac,e
    !real(real64),allocatable::e(:,:,:)
    integer(kind=int64) :: i,j,k,ip1,jp1,kp1
   
    !allocate(e(ni,nj,nk))
    !e(:,:,:) = 0.0d0
    e = 0.0d0
    do concurrent(k=1:nk,j=1:nj,i=1:ni) local(ip1,jp1,kp1,ddot,crossx,crossy,crossz,sfac) reduce(+:e)
       ip1 = mod(i,ni) + 1
       jp1 = mod(j,nj) + 1
       kp1 = k+1
       ddot = nx(i,j,k)*nx(ip1,j,k) + ny(i,j,k)*ny(ip1,j,k)+nz(i,j,k)*nz(ip1,j,k)
       crossx = ny(i,j,k)*nz(ip1,j,k)-nz(i,j,k)*ny(ip1,j,k)
       sfac = 0.5d0*(s(i,j,k)+s(ip1,j,k))
       e = e + (1.0d0 - ddot*ddot) - KK*ddot*crossx*sfac

       ddot = nx(i,j,k)*nx(ip1,jp1,k) + ny(i,j,k)*ny(i,jp1,k)+nz(i,j,k)*nz(i,jp1,k)
       crossy = nz(i,j,k)*nx(i,jp1,k)-nx(i,j,k)*nz(i,jp1,k)
       sfac = 0.5d0*(s(i,j,k)+s(i,jp1,k))
       e = e + (1.0d0 - ddot*ddot) - KK*ddot*crossy*sfac
       
       if (kp1 .le. nk) then
       ddot = nx(i,j,k)*nx(i,j,kp1) + ny(i,j,k)*ny(i,j,kp1)+nz(i,j,k)*nz(i,j,kp1)
       crossz = nx(i,j,k)*ny(i,j,kp1) - ny(i,j,k)*nx(i,j,kp1)
       sfac = 0.5d0*(s(i,j,k)+s(i,j,kp1))
       e = e + (1.0d0 - ddot*ddot) - KK*ddot*crossz*sfac
       endif
    enddo
    etot = e/float(ni*nj*nk)
    !etot = sum(e)/float(ni*nj*nk)
    !deallocate(e)    
  end function etot

end module chll_mod
