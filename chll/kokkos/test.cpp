//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include <Kokkos_Core.hpp>

void slinit(auto& sl1, auto& sl2,int ni,int nj,int nk);
	
int main( int argc, char* argv[] )
{
  int ni = 64;
  int nj = 64;
  int nk = 64;
  int n3 = ni*nj*nk;
  int nsub = n3/2;
  int nsteps = 1000;
  double kbt = 0.05;
  double d = 0.01;
  double KK = 0.0;
  double rho = 0.0;
  double g = 0.0;
  Kokkos::initialize( argc, argv );
  {

  #ifdef KOKKOS_ENABLE_CUDA
  #define MemSpace Kokkos::CudaSpace
  #endif
  #ifdef KOKKOS_ENABLE_HIP
  #define MemSpace Kokkos::Experimental::HIPSpace
  #endif
  #ifdef KOKKOS_ENABLE_OPENMPTARGET
  #define MemSpace Kokkos::OpenMPTargetSpace
  #endif

  #ifndef MemSpace
  #define MemSpace Kokkos::HostSpace
  #endif

  using ExecSpace = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;
  typedef Kokkos::MDRangePolicy< Kokkos::Rank<3> > mdrange_policy;
  // Allocate y, x vectors and Matrix A on device.
  typedef Kokkos::View<double*, Kokkos::LayoutLeft, MemSpace>   ViewVectorTypeDouble;
  typedef Kokkos::View<double***, Kokkos::LayoutLeft, MemSpace>  ViewMatrixTypeDouble;
  typedef Kokkos::View<int***, Kokkos::LayoutLeft, MemSpace> ViewMatrixTypeInt;
  typedef Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace> ViewVectorTypeInt;
  ViewMatrixTypeDouble nx( "nx", ni,nj,nk);
  ViewMatrixTypeDouble ny( "ny", ni,nj,nk);
  ViewMatrixTypeDouble nz( "nz", ni,nj,nk);
  ViewMatrixTypeInt s( "s" ,ni,nj,nk);
  ViewMatrixTypeInt dope("dope",ni,nj,nk);
  ViewVectorTypeInt sl1("sl1",nsub);
  ViewVectorTypeInt sl2("sl2",nsub);
  
  // Create host mirrors of device views.
  ViewMatrixTypeDouble::HostMirror h_nx = Kokkos::create_mirror_view( nx );
  ViewMatrixTypeDouble::HostMirror h_ny = Kokkos::create_mirror_view( ny );
  ViewMatrixTypeDouble::HostMirror h_nz = Kokkos::create_mirror_view( nz );
  ViewMatrixTypeInt::HostMirror h_s = Kokkos::create_mirror_view( s );
  ViewMatrixTypeInt::HostMirror h_dope = Kokkos::create_mirror_view( dope );
  ViewVectorTypeInt::HostMirror h_sl1 = Kokkos::create_mirror_view( sl1 );
  ViewVectorTypeInt::HostMirror h_sl2 = Kokkos::create_mirror_view( sl2 );

  // Initialize on host.
  for ( int k = 0; k < nk; ++k) {
    for ( int j = 0; j < nj; ++j ) {
      for ( int i = 0; i < ni; ++i ) {
        h_s(k, j, i ) = 1;
        if (rand() < 0.5) {
			h_s(k,j,i) = -1;
		}
		h_dope(k,j,i) = 0;
		h_nx(k,j,i) = 1.0;
		h_ny(k,j,i) = 0.0;
		h_nz(k,j,i) = 0.0;
      }
    }
  }
  slinit(h_sl1,h_sl1,ni,nj,nk);
  // Deep copy host views to device views.
  Kokkos::deep_copy( nx, h_nx );
  Kokkos::deep_copy( ny, h_ny );
  Kokkos::deep_copy( nz, h_nz );
  Kokkos::deep_copy( s, h_s);
  Kokkos::deep_copy( dope, h_dope);
  Kokkos::deep_copy( sl1, h_sl1);
  Kokkos::deep_copy( sl2, h_sl2);

  Kokkos::fence();
  int nstart = 1;
  int nout = 50;
  run(nstart,nstop,nout,nx,ny,nz,s,dope,sl1,sl2,g,KK,d,kbt,ni,nj,nk,nsub,msteps,m);
  /*

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    Kokkos::parallel_reduce( "yAx", range_policy( 0, N ), KOKKOS_LAMBDA ( int j, double &update ) {
      double temp2 = 0;

      for ( int i = 0; i < M; ++i ) {
        temp2 += A( j, i ) * x( i );
      }

      update += y( j ) * temp2;
    }, result );

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d is %lf\n", N, M, result );
    }

    const double solution = (double) N * (double) M;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );
  */
  }
  
  Kokkos::finalize();
  printf("Done!\n");

  return 0;
}
void run(int nstart,int nstop,int nout,auto &nx, auto &ny,auto &nz,auto &s, auto &dope,auto &sl1,auto &sl2,double g,double KK,double d,double kbt,int ni,int nj, int nk,int nsub) {
	int naccept = 0;
	int nflip = 0;
	for (int istep = nstart, istep <= nstop; ++istep) {
		evolve(sl1,nx,ny,nz,s,g,KK,d,kbt,nsub,naccept,nflip,ni,nj,nk);
		Kokkos::fence();
		evolve(sl2,nx,ny,nz,s,g,KK,d,kbt,nsub,naccept,nflip,ni,nj,nk);
		Kokkos::fence();
		paccept = double(naccept)/double(n3);
		
	}
	
}


void slinit(auto& sl1, auto& sl2,int ni,int nj,int nk) {
	/* 	  ! create list of site indices for each sublattice, sub1, sub2
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
	  enddo */
	  printf("> SL Init\n");
	  int nsub1 = 0;
	  int nsub2 = 0;
	  for ( int k = 0; k < nk; ++k) {
		  for ( int j = 0; j < nj; ++j ) {
			  for ( int i = 0; i < ni; ++i ) {
				  int idx = i + j * ni + k*ni*nj;
				  if ((i+j+k)%2 != 0) {
					  sl1(nsub1) = idx;
					  nsub1 += 1;
				  } else {
					  sl2(nsub2) = idx;
					  nsub2 += 1;
				  }
			  }
		  }
	  }
}

