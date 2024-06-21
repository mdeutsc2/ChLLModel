#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define REAL double
#define INT int64_t

void init(INT ni, INT nj, INT nk, REAL nx[ni][nj][nk], REAL ny[ni][nj][nk], REAL nz[ni][nj][nk],
          INT s[ni][nj][nk], INT dope[ni][nj][nk], INT sl1[], INT sl2[])
{
    REAL rnd, costh, sinth, phi, cosphi, sinphi, pi, twopi;
    INT i, j, k, nsub1, nsub2, idx;

    pi = 4.0 * atan(1.0);
    twopi = 2.0 * pi;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            for (k = 0; k < nk; k++)
            {
                s[i][j][k] = 1;
                rnd = (REAL)rand() / RAND_MAX;
                if (rnd <= 0.5)
                    s[i][j][k] = -1;

                dope[i][j][k] = 0;
                rnd = (REAL)rand() / RAND_MAX;
                costh = 2.0 * (rnd - 0.5);
                sinth = sqrt(1.0 - costh * costh);
                rnd = (REAL)rand() / RAND_MAX;
                phi = rnd * twopi;
                cosphi = cos(phi);
                sinphi = sin(phi);
                nx[i][j][k] = sinth * cosphi;
                ny[i][j][k] = sinth * sinphi;
                nz[i][j][k] = costh;
            }
        }
    }

    nsub1 = 0;
    nsub2 = 0;
    for (k = 0; k < nk; k++)
    {
        for (j = 0; j < nj; j++)
        {
            for (i = 0; i < ni; i++)
            {
                idx = i + j * ni + k * (ni * nj);
                if ((i + j + k) % 2 != 0)
                {
                    sl1[nsub1] = idx;
                    nsub1++;
                }
                else
                {
                    sl2[nsub2] = idx;
                    nsub2++;
                }
            }
        }
    }
}

void run(INT nsteps, INT ni, INT nj, INT nk, REAL nx[ni][nj][nk], REAL ny[ni][nj][nk], REAL nz[ni][nj][nk],
         INT s[ni][nj][nk], INT dope[ni][nj][nk], INT sl1[], INT sl2[],
         REAL rand1[ni*nj*nk][2], REAL rand2[ni*nj*nk][2], REAL KK, REAL d, REAL kbt)
{
    INT naccept, nflip, istep, nsub, n3;
    REAL paccept, faccept, total_energy, e_excess;

    n3 = ni * nj * nk;
    nsub = n3 / 2;

    for (istep = 1; istep <= nsteps; istep++)
    {
        naccept = 0;
        nflip = 0;

        for (int itry = 0; itry < n3; itry++)
        {
            rand1[itry][0] = (REAL)rand() / RAND_MAX;
            rand2[itry][0] = (REAL)rand() / RAND_MAX;
            evolve(sl1, ni, nj, nk, nx, ny, nz, s, rand1, rand2, KK, d, kbt, nsub, &naccept, &nflip);

            rand1[itry][0] = (REAL)rand() / RAND_MAX;
            rand2[itry][0] = (REAL)rand() / RAND_MAX;
            evolve(sl2, ni, nj, nk, nx, ny, nz, s, rand1, rand2, KK, d, kbt, nsub, &naccept, &nflip);
        }

        paccept = (REAL)naccept / (REAL)n3;
        if (paccept < 0.4)
            d *= 0.995;
        else if (paccept > 0.6)
            d /= 0.995;

        if (istep % 100 == 0)
        {
            faccept = (REAL)nflip / (REAL)n3;
            total_energy = etot(nx, ny, nz, s, KK, ni, nj, nk);
            e_excess = 0.0; // Calculate this if needed
            printf("%ld %f %f %f %f\n", (long)istep, total_energy, e_excess, paccept, faccept);
        }
    }
}

void evolve(INT sl[], INT ni, INT nj, INT nk, REAL nx[ni][nj][nk], REAL ny[ni][nj][nk], REAL nz[ni][nj][nk],
            INT s[ni][nj][nk], REAL rand1[ni*nj*nk][2], REAL rand2[ni*nj*nk][2], REAL KK, REAL d, REAL kbt,
            INT nsub, INT *naccept, INT *nflip)
{
    INT itry, ip1, im1, jp1, jm1, kp1, km1, snew, index;
    REAL dcosphi, dsinphi, enew, eold, nnx, nny, nnz, ux, uy, uz, vx, vy, vz, xxnew, yynew, zznew, rsq, phi;
    REAL dott, crossx, crossy, crossz, sfac, pi, twopi;

    pi = 4.0 * atan(1.0);
    twopi = 2.0 * pi;

    for (itry = 0; itry < nsub; itry++)
    {
        index = sl[itry];
        int idx = index - 1;
        int i = 1 + (idx % ni);
        int j = 1 + (idx / ni) % nj;
        int k = 1 + (idx / (ni * nj));

        ip1 = i % ni + 1;
        im1 = (i - 2 + ni) % ni + 1;
        jp1 = j % nj + 1;
        jm1 = (j - 2 + nj) % nj + 1;
        kp1 = k + 1;
        km1 = k - 1;

        nnx = nx[i-1][j-1][k-1];
        nny = ny[i-1][j-1][k-1];
        nnz = nz[i-1][j-1][k-1];
        eold = energy(nx, ny, nz, nnx, nny, nnz, s[i-1][j-1][k-1], s, KK, i, j, k, ip1, im1, jp1, jm1, kp1, km1, ni, nj, nk);

        if (fabs(nnz) > 0.999)
        {
            phi = rand1[itry][0] * twopi;
            xxnew = nnx + d * cos(phi);
            yynew = nny + d * sin(phi);
            zznew = nnz;
            rsq = sqrt(xxnew * xxnew + yynew * yynew + zznew * zznew);
            nnx = xxnew / rsq;
            nny = yynew / rsq;
            nnz = zznew / rsq;
        }
        else
        {
            ux = -nny;
            uy = nnx;
            uz = 0.0;
            rsq = sqrt(ux * ux + uy * uy);
            ux /= rsq;
            uy /= rsq;
            uz /= rsq;

            vx = -nnz * nnx;
            vy = -nnz * nny;
            vz = nnx * nnx + nny * nny;
            rsq = sqrt(vx * vx + vy * vy + vz * vz);
            vx /= rsq;
            vy /= rsq;
            vz /= rsq;

            phi = rand1[itry][1] * twopi;
            dcosphi = d * cos(phi);
            dsinphi = d * sin(phi);
            nnx = nnx + dcosphi * ux + dsinphi * vx;
            nny = nny + dcosphi * uy + dsinphi * vy;
            nnz = nnz + dcosphi * uz + dsinphi * vz;
            rsq = sqrt(nnx * nnx + nny * nny + nnz * nnz);
            nnx /= rsq;
            nny /= rsq;
            nnz /= rsq;
        }

        enew = energy(nx, ny, nz, nnx, nny, nnz, s[i-1][j-1][k-1], s, KK, i, j, k, ip1, im1, jp1, jm1, kp1, km1, ni, nj, nk);

        if (rand2[itry][0] < exp(-1.0 / kbt * (enew - eold)))
        {
            *naccept = *naccept + 1;
            s[i-1][j-1][k-1] = -s[i-1][j-1][k-1];
        }
    }
}

REAL energy(REAL nx[ni][nj][nk], REAL ny[ni][nj][nk], REAL nz[ni][nj][nk],
            REAL nnx, REAL nny, REAL nnz, INT sijk, INT s[ni][nj][nk], REAL KK,
            INT i, INT j, INT k, INT ip1, INT im1, INT jp1, INT jm1, INT kp1, INT km1, INT ni, INT nj, INT nk)
{
    INT ijk, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    REAL ee;

    iim1 = (i - 2 + ni) % ni + 1;
    iip1 = i % ni + 1;
    jjm1 = (j - 2 + nj) % nj + 1;
    jjp1 = j % nj + 1;
    kkm1 = k - 1;
    kkp1 = k + 1;

    ee = -0.5 * KK * (s[i-1][j-1][k-1] * s[iip1-1][j-1][k-1] + s[i-1][j-1][k-1] * s[iim1-1][j-1][k-1] +
                     s[i-1][j-1][k-1] * s[i-1][jjp1-1][k-1] + s[i-1][j-1][k-1] * s[i-1][jjm1-1][k-1] +
                     s[i-1][j-1][k-1] * s[i-1][j-1][kkp1-1] + s[i-1][j-1][k-1] * s[i-1][j-1][kkm1-1]);
    ee = ee - nnx * nx[i-1][j-1][k-1] - nny * ny[i-1][j-1][k-1] - nnz * nz[i-1][j-1][k-1];

    return ee;
}

REAL etot(REAL nx[ni][nj][nk], REAL ny[ni][nj][nk], REAL nz[ni][nj][nk],
          INT s[ni][nj][nk], REAL KK, INT ni, INT nj, INT nk)
{
    INT i, j, k, iip1, iim1, jjp1, jjm1, kkp1, kkm1;
    REAL ee, total_energy;

    total_energy = 0.0;

    for (k = 0; k < nk; k++)
    {
        for (j = 0; j < nj; j++)
        {
            for (i = 0; i < ni; i++)
            {
                iim1 = (i - 2 + ni) % ni + 1;
                iip1 = i % ni + 1;
                jjm1 = (j - 2 + nj) % nj + 1;
                jjp1 = j % nj + 1;
                kkm1 = k - 1;
                kkp1 = k + 1;

                ee = -0.5 * KK * (s[i][j][k] * s[iip1][j][k] + s[i][j][k] * s[iim1][j][k] +
                                 s[i][j][k] * s[i][jjp1][k] + s[i][j][k] * s[i][jjm1][k] +
                                 s[i][j][k] * s[i][j][kkp1] + s[i][j][k] * s[i][j][kkm1]);
                ee = ee - nx[i][j][k] * nx[i][j][k] - ny[i][j][k] * ny[i][j][k] - nz[i][j][k] * nz[i][j][k];

                total_energy += ee;
            }
        }
    }

    return total_energy;
}

