import warp as wp

### FUNCTIONS
@wp.func
def energy(nx: wp.array3d(dtype=wp.float64),
           ny: wp.array3d(dtype=wp.float64),
           nz: wp.array3d(dtype=wp.float64),
           nnx: wp.float64,
           nny: wp.float64,
           nnz: wp.float64,
           snew: wp.int32,
           sold: wp.array3d(dtype=wp.int32),
           KK: wp.float64,
           i:wp.int32,
           j:wp.int32,
           k:wp.int32,
           ip1: wp.int32,
           im1: wp.int32,
           jp1: wp.int32,
           jm1: wp.int32,
           kp1: wp.int32,
           km1:wp.int32,
           ni:wp.int32,
           nj:wp.int32,
           nk:wp.int32):
    e = wp.float64(0.0)
    if (ip1 <= ni-1):
        dott = nnx*nx[ip1,j,k] + nny*ny[ip1,j,k] + nnz*nz[ip1,j,k]
        crossx = nny*nz[ip1,j,k] - nnz*ny[ip1,j,k]
        sfac = wp.float64(0.5)*wp.float64((snew + sold[ip1,j,k]))
        e += (wp.float64(1.0) - dott*dott) - KK*dott*crossx*sfac
    if (im1 >= 0):
        dott = nnx*nx[im1,j,k] + nny*ny[im1,j,k] + nnz*nz[im1,j,k]
        crossx = nny*nz[im1,j,k] - nnz*ny[im1,j,k]
        sfac = wp.float64(0.5)*wp.float64((snew + sold[im1,j,k]))
        e += (wp.float64(1.0) - dott*dott) + KK*dott*crossx*sfac
    if (jp1 <= nj-1):
        dott = nnx*nx[i,jp1,k] + nny*ny[i,jp1,k] + nnz*nz[i,jp1,k]
        crossy = nnz*nx[i,jp1,k] - nnx*nz[i,jp1,k]
        sfac = wp.float64(0.5)*wp.float64(snew + sold[i,jp1,k])
        e += (wp.float64(1.0) - dott*dott) - KK*dott*crossy*sfac
    if (jm1 >= 0):
        dott = nnx*nx[i,jm1,k] + nny*ny[i,jm1,k] + nnz*nz[i,jm1,k]
        crossy = nnz*nx[i,jm1,k] - nnx*nz[i,jm1,k]
        sfac = wp.float64(0.5)*wp.float64(snew + sold[i,jm1,k])
        e += (wp.float64(1.0) - dott*dott) + KK*dott*crossy*sfac
    if (kp1 <= nk-1):
        dott = nnx*nx[i,j,kp1] + nny*ny[i,j,kp1] + nnz*nz[i,j,kp1]
        crossz = nnx*ny[i,j,kp1] - nny*nx[i,j,kp1]
        sfac = wp.float64(0.5)*wp.float64(snew + sold[i,k,kp1])
        e += (wp.float64(1.0) - dott*dott) - KK*dott*crossz*sfac
    if (km1 >= 0):
        dott = nnx*nx[i,j,km1] + nny*ny[i,j,km1] + nnz*nz[i,j,km1]
        crossz = nnz*nx[i,j,km1] - nnx*nz[i,j,km1]
        sfac = wp.float64(0.5)*wp.float64(snew + sold[i,j,km1])
        e += (wp.float64(1.0) - dott*dott) + KK*dott*crossz*sfac
    return e
        

### KERNELS
@wp.kernel
def reset_counters(naccept: wp.array3d(dtype=wp.int32), nflip: wp.array3d(dtype=wp.int32)):
    i,j,k = wp.tid()
    naccept[i,j,k] = 0
    nflip[i,j,k] = 0

@wp.kernel
def evolve(sl: wp.array(dtype=wp.int32),
           nx: wp.array3d(dtype=wp.float64),
           ny: wp.array3d(dtype=wp.float64),
           nz: wp.array3d(dtype=wp.float64),
           s: wp.array3d(dtype=wp.int32),
           naccept: wp.array3d(dtype=wp.int32),
           nflip: wp.array3d(dtype=wp.int32),
           kbt: wp.float64,
           KK: wp.float64,
           d:wp.float64,
           ni:wp.int32,nj:wp.int32,nk:wp.int32,seed:wp.int32):
    tid = wp.tid()
    rng = wp.rand_init(seed,tid)
    index = sl[tid]
    k = index % nk
    j = (index // nk) % nj
    i = index // (nj*nk)

    ip1 = i + 1
    im1 = i - 1
    jp1 = j + 1
    jm1 = j - 1
    kp1 = k + 1
    km1 = k - 1

    nnx = nx[i,j,k]
    nny = ny[i,j,k]
    nnz = nz[i,j,k]

    eold = energy(nx,ny,nz,nnx,nny,nnz,s[i,j,k],s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
    #nnx,nny,nnz = get_perturb(nnx,nny,nnz,d,rng)
    if (wp.abs(nnz) > 0.999):
        phi = wp.randf(rng)*8.0*wp.atan(1.0) #twopi
        xxnew = nnx+d*wp.float64(wp.cos(phi))
        yynew = nny+d*wp.float64(wp.sin(phi))
        zznew = nnz
        rsq = wp.sqrt(xxnew*xxnew + yynew*yynew + zznew*zznew)
        nnx = xxnew/rsq
        nny = yynew/rsq
        nwz = zznew/rsq
    else:
        ux = -nny
        uy = nnx
        uz = wp.float64(0.0)
        rsq = wp.sqrt(ux*ux+uy*uy+uz*uz)
        ux = ux/rsq
        uy = uy/rsq
        uz = uz/rsq
        vx = -nnz*nnx
        vy = -nnz*nny
        vz = nnx*nnx + nny*nny
        rsq = wp.sqrt(vx*vx + vy*vy + vz*vz)
        vx = vx/rsq
        vy = vy/rsq
        vz = vz/rsq
        phi = wp.randf(rng)*8.0*wp.atan(1.0) # twopi
        dcosphi = d*wp.float64(wp.cos(phi))
        dsinphi = d*wp.float64(wp.sin(phi))
        newx = nnx+dcosphi*ux + dsinphi*vx
        newy = nny+dcosphi*uy + dsinphi*vy
        newz = nnz+dcosphi*uz + dsinphi*vz
        rsq = wp.sqrt(newx*newx + newy*newy + newz*newz)
        nnx = newx/rsq
        nny = newy/rsq
        nnz = newz/rsq
    enew = energy(nx,ny,nz,nnx,nny,nnz,s[i,j,k],s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
    if (enew < eold):
        naccept[i,j,k] += 1
        nx[i,j,k] = nnx
        ny[i,j,k] = nny
        nz[i,j,k] = nnz
    else:
        if (wp.randf(rng) <= wp.exp(-(enew-eold)/kbt)):
            nx[i,j,k] = nnx
            ny[i,j,k] = nny
            nz[i,j,k] = nnz
            naccept[i,j,k] = naccept[i,j,k] + 1
    #print("145")
    
    # chirality step
    nnx = nx[i,j,k]
    nny = ny[i,j,k]
    nnz = nz[i,j,k]

    eold = energy(nx,ny,nz,nnx,nny,nnz,s[i,j,k],s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
    snew = -s[i,j,k]
    enew = energy(nx,ny,nz,nnx,nny,nnz,snew,s,KK,i,j,k,ip1,im1,jp1,jm1,kp1,km1,ni,nj,nk)
    if (enew < eold):
        s[i,j,k] = snew
        nflip[i,j,k] = nflip[i,j,k] + 1
    else:
        if (wp.randf(rng) <= wp.exp(-(enew-eold)/kbt)):
            s[i,j,k] = snew
            nflip[i,j,k] = nflip[i,j,k] + 1

@wp.kernel
def etot(e: wp.array3d(dtype=wp.float64),
         nx:wp.array3d(dtype=wp.float64),
         ny:wp.array3d(dtype=wp.float64),
         nz:wp.array3d(dtype=wp.float64),
         s:wp.array3d(dtype=wp.int32),
         KK: wp.float64,ni:wp.int32,nj:wp.int32,nk:wp.int32):
    i,j,k = wp.tid()
    ip1 = i%(ni-1) + 1
    jp1 = j%(nj-1) + 1
    kp1 = k + 1
    e[i,j,k] = wp.float64(0.0)
    ddot = nx[i,j,k]*nx[ip1,j,k] + ny[i,j,k]*ny[ip1,j,k] + nz[i,j,k]*nz[ip1,j,k]
    crossx = ny[i,j,k]*nz[ip1,j,k] - nz[i,j,k]*ny[ip1,j,k]
    sfac = wp.float64(0.5)*wp.float64(s[i,j,k]+s[ip1,j,k])
    e[i,j,k] = e[i,j,k] + (wp.float64(1.0) - ddot*ddot) - KK*ddot*crossx*sfac

    ddot = nx[i,j,k]*nx[i,jp1,k] + ny[i,j,k]*ny[i,jp1,k] + nz[i,j,k]*nz[i,jp1,k]
    crossy = nz[i,j,k]*nx[i,jp1,k] - nx[i,j,k]*nz[i,jp1,k]
    sfac = wp.float64(0.5)*wp.float64(s[i,j,k]+s[i,jp1,k])
    e[i,j,k] = e[i,j,k] + (wp.float64(1.0) - ddot*ddot) - KK*ddot*crossx*sfac

    if (kp1 <= nk-1):
        ddot = nx[i,j,k]*nx[i,j,kp1] + ny[i,j,k]*ny[i,j,kp1] + nz[i,j,k]*nz[i,j,kp1]
        crossx = nx[i,j,k]*ny[i,j,kp1] - ny[i,j,k]*nx[i,j,kp1]
        sfac = wp.float64(0.5)*wp.float64(s[i,j,k]+s[i,j,kp1])
        e[i,j,k] = e[i,j,k] + (wp.float64(1.0) - ddot*ddot) - KK*ddot*crossx*sfac
