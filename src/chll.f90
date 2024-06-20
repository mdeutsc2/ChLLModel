program chll
  use types
  use chll_mod
  !use omp_lib
   implicit none
  
  integer(kind=int64) :: ni,nj,nk,n3
  real(kind=real64) :: kbt = 0.05d0
  integer(kind=int64) :: nsteps = 1000
  real(kind=real64),allocatable :: nx(:,:,:),ny(:,:,:),nz(:,:,:)
  integer(kind=int64),allocatable :: s(:,:,:) ! chirality
  integer(kind=int64),allocatable :: sl1(:),sl2(:) ! sublattices corresponding to odds(1) and evens(2)
  integer(kind=int64),allocatable :: dope(:,:,:)!naccept(:,:,:),nflip(:,:,:)
  real(kind=real64),allocatable :: rand1(:,:),rand2(:,:)
  real(kind=real64) :: d = 0.01d0 ! size of spin rotation perturbation
  real(kind=real64) :: KK = 1.d0 !
  real(kind=real64) :: rho=0.0d0 ! density of chiral dopant
  real(kind=real64),allocatable :: rhoz(:) !measuring density of chiral in each z-slice
  real(kind=real64) :: cosphi,sinphi,costh,sinth,phi,pi,twopi,rnd
  real(kind=real64) :: faccept,e_excess,paccept,total_energy
  integer(kind=int64) :: nsub,nsub1,nsub2,index
  integer(kind=int64) :: i,j,k,istep,itry,naccept,nflip
  real(kind=real64) :: scale,x1,x2,z1,z2 ! for output

  ni = 64
  nj = 64
  nk = 64
  n3 = ni*nj*nk
  pi = 4.0d0*datan(1.0d0)
  twopi = 2.0d0*pi
  nsub = ni*nj*nk/2

  allocate(nx(ni,nj,nk))
  allocate(ny(ni,nj,nk))
  allocate(nz(ni,nj,nk))
  allocate(s(ni,nj,nk))
  allocate(dope(ni,nj,nk))
  !allocate(naccept(ni,nj,nk))
  !allocate(nflip(ni,nj,nk))
  allocate(sl1(nsub))
  allocate(sl2(nsub))
  allocate(rand1(nsub,2))
  allocate(rand2(nsub,2))
  call init(nx,ny,nz,s,dope,sl1,sl2,ni,nj,nk)
  print*,"init done!"
  call run(nsteps,nx,ny,nz,s,dope,sl1,sl2,rand1,rand2,KK,d,kbt,ni,nj,nk)
  call output(nx,ny,nz,s,ni,nj,nk)
  !deallocate(nx,ny,nz,nflip,naccept,s,dope)
  deallocate(nx,ny,nz,s,dope)
  
contains
	subroutine output(nx,ny,nz,s,ni,nj,nk)
		implicit none
		real(kind=real64), intent(in out) :: nx(:,:,:),ny(:,:,:),nz(:,:,:)
		integer(kind=int64), intent(in out) :: s(:,:,:)
		integer(kind=int64), intent(in) :: ni,nj,nk
		integer(kind=int64) :: i,j,k
		real(kind=real64) :: x1,x2,z1,z2,scale
		open(unit=11,file='LLMC-configa.dat',status='unknown')
		open(unit=12,file='LLMC-configb.dat',status='unknown')
		scale = 0.4
		do i = 1,ni
		 j = nj/2
		 do k = 1,nk
			x1 = i-scale*nx(i,j,k)
			x2 = i+scale*nx(i,j,k)
			z1 = k-scale*nz(i,j,k)
			z2 = k+scale*nz(i,j,k)
			if (s(i,j,k).eq.1) then
			   write(11,*) x1,z1
			   write(11,*) x2,z2
			   write(11,*)
			else
			   write(12,*) x1,z1
			   write(12,*) x2,z2
			   write(12,*)
			endif
		 enddo
		enddo
		close(unit=11)
		close(unit=12)
	end subroutine output
     
end program chll
