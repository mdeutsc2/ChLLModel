program chll
  use iso_c_binding
  use chll_mod
  !use omp_lib
   implicit none
  
  integer(c_int) :: ni,nj,nk,n3
  real(c_double) :: kbt = 0.05d0
  integer(c_int) :: nsteps = 1000
  real(c_double),allocatable :: nx(:,:,:),ny(:,:,:),nz(:,:,:)
  integer(c_int),allocatable :: s(:,:,:) ! chirality
  integer(c_int),allocatable :: sl1(:),sl2(:) ! sublattices corresponding to odds(1) and evens(2)
  integer(c_int),allocatable :: dope(:,:,:)!naccept(:,:,:),nflip(:,:,:)
  real(c_double) :: d = 0.01d0 ! size of spin rotation perturbation
  real(c_double) :: KK = 1.0d0 !
  real(c_double) :: rho=0.0d0 ! density of chiral dopant
  real(c_double) :: g = 0.0d0 ! 
  real(c_double),allocatable :: rhoz(:) !measuring density of chiral in each z-slice
  real(c_double) :: cosphi,sinphi,costh,sinth,phi,pi,twopi,rnd
  real(c_double) :: faccept,e_excess,paccept,total_energy
  integer(c_int) :: nsub,nsub1,nsub2,index
  integer(c_int) :: i,j,k,istep,itry,naccept,nflip
  real(c_double) :: scale,x1,x2,z1,z2 ! for output

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
  
  do i = 1,ni
  do j = 1,nj
  do k = 1,nk
	s(i,j,k) = 1
	call random_number(rnd)
	if (rnd.le.0.5d0) s(i,j,k) = -1
	dope(i,j,k) = 0
	nx(i,j,k) = 1.0d0
	ny(i,j,k) = 0.0d0
	nz(i,j,k) = 0.0d0
  enddo 
  enddo
  enddo
	
  call init(nx,ny,nz,s,dope,sl1,sl2,ni,nj,nk,nsub)
  print*,"init done!"
  call run(1,nsteps,nx,ny,nz,s,dope,sl1,sl2,g,KK,d,kbt,ni,nj,nk,nsub)
  !call output(nx,ny,nz,s,ni,nj,nk)
  !deallocate(nx,ny,nz,nflip,naccept,s,dope)
  !deallocate(nx,ny,nz,s,dope)

     
end program chll
