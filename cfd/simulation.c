#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"
#include <omp.h>
#include "mpi.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int *ileft, *iright;
extern int nprocs, proc;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re)
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;
    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int initial_imax,end_imax,total_imax_iteration_by_rank;
    total_imax_iteration_by_rank = floor(imax/size);
    initial_imax = rank*total_imax_iteration_by_rank;
    end_imax = initial_imax + total_imax_iteration_by_rank;
    initial_imax++;

    int displacements[size];
    int counts[size];

    for (int index = 0; index < size; index++)
    {
        displacements[index] = 0;
        counts[index] = total_imax_iteration_by_rank*(jmax+2);
        if(imax%size !=0){
            counts[size-1] = (total_imax_iteration_by_rank + (imax%size))*(jmax+2);
        }
    }

    for (int index = 1; index < size; index++)
    {
        displacements[index] = displacements[index-1] + total_imax_iteration_by_rank*(jmax+2);
    }

    if( (rank == size -1) && (imax%size != 0)){ 
            total_imax_iteration_by_rank += (imax%size);
            end_imax = end_imax + (imax%size);
    }
 
    if (rank == (size-1)){end_imax = end_imax -1;} 

    for (i=initial_imax; i<=end_imax; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                    gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                    (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                    gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                    /(4.0*delx);
                duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                    gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                    (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                    gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                    /(4.0*dely);
                laplu = (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx+
                    (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely;
   
                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    MPI_Allgatherv(&f[initial_imax][1], total_imax_iteration_by_rank*(jmax+2), MPI_FLOAT, &f[1][1], counts,displacements, MPI_FLOAT, MPI_COMM_WORLD);


    if (rank == (size-1)){end_imax = end_imax+1;} 

    for (i=initial_imax; i<=end_imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                    gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                    (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                    gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                    /(4.0*delx);
                dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                    gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                    (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                    gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                    /(4.0*dely);

                laplv = (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx+
                    (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely;

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    MPI_Allgatherv(&g[initial_imax][1], total_imax_iteration_by_rank*(jmax+2), MPI_FLOAT, &g[1][1], counts,displacements, MPI_FLOAT, MPI_COMM_WORLD);

    /* f & g at external boundaries */
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    
    for (i=1; i<=imax; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
    MPI_Barrier(MPI_COMM_WORLD );
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely)
{
    int i, j;
    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int initial_imax,end_imax,total_imax_iteration_by_rank;
    total_imax_iteration_by_rank = floor(imax/size);
    initial_imax = rank*total_imax_iteration_by_rank;
    end_imax = initial_imax + total_imax_iteration_by_rank;
    initial_imax++;

    int displacements[size];
    int counts[size];

    for (int index = 0; index < size; index++)
    {
        displacements[index] = 0;
        counts[index] = total_imax_iteration_by_rank*(jmax+2);
        if(imax%size !=0){
            counts[size-1] = (total_imax_iteration_by_rank + (imax%size))*(jmax+2);
        }
    }

    for (int index = 1; index < size; index++)
    {
        displacements[index] = displacements[index-1] + total_imax_iteration_by_rank*(jmax+2);
    }

    if( (rank == size -1) && (imax%size != 0)){ 
            total_imax_iteration_by_rank += (imax%size);
            end_imax = end_imax + (imax%size);
    }
    
    for (i=initial_imax;i<=end_imax;i++) {
        for (j=1;j<=jmax;j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = (
                             (f[i][j]-f[i-1][j])/delx +
                             (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
            }
        }
    }
    MPI_Allgatherv(&rhs[initial_imax][1], total_imax_iteration_by_rank*(jmax+2), MPI_FLOAT, &rhs[1][1], counts,displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD );
}


/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull)
{
    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int initial_imax,end_imax,total_imax_iteration_by_rank;
    total_imax_iteration_by_rank = floor(imax/size);
    initial_imax = rank*total_imax_iteration_by_rank;
    end_imax = initial_imax + total_imax_iteration_by_rank;
    initial_imax++;

    int displacements[size];
    int counts[size];

    for (int index = 0; index < size; index++)
    {
        displacements[index] = 0;
        counts[index] = total_imax_iteration_by_rank*(jmax+2);
        if(imax%size !=0){
            counts[size-1] = (total_imax_iteration_by_rank + (imax%size))*(jmax+2);
        }
    }

    for (int index = 1; index < size; index++)
    {
        displacements[index] = displacements[index-1] + total_imax_iteration_by_rank*(jmax+2);
    }

    if( (rank == size -1) && (imax%size != 0)){ 
            total_imax_iteration_by_rank += (imax%size);
            end_imax = end_imax + (imax%size);
    }
    

    
    // printf("initial %d end %d total iteration %d rank %d size %d \n",initial_imax,end_imax,total_imax_iteration_by_rank,rank,size);
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;

    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    int rankL = rank > 0        ? rank - 1 : MPI_PROC_NULL;
    int rankR = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    
    /* Calculate sum of squares */
    for (i = initial_imax; i <= end_imax; i++) {
        // #pragma omp parallel for schedule(static) num_threads(2) reduction(+:p0)
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }

    MPI_Allreduce( &p0, &p0, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );

    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
            // if want to test with more larger problem size please un-comment the syntax below
            // #pragma omp parallel for schedule(static) num_threads(4) 
            for (i = initial_imax; i <= end_imax; i++) {
                for (j = 1; j <= jmax; j++) {  
                    if ((i+j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1.-omega)*p[i][j] - 
                              beta_2*(
                                    (p[i+1][j]+p[i-1][j])*rdx2
                                  + (p[i][j+1]+p[i][j-1])*rdy2
                                  -  rhs[i][j]
                              );
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p[i][j] = (1.-omega)*p[i][j] -
                            beta_mod*(
                                  (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                                + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                    }
                } /* end of j */
            } /* end of i */

            //All boundaries p will be send to adjacent processors
            // Use the rank of the sending processor as the tag
            MPI_Sendrecv(&p[initial_imax][1], jmax, MPI_FLOAT, rankL, 0,             // sending to left
                            &p[initial_imax-1][1], jmax, MPI_FLOAT, rankL, 0,            // receiving from left 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&p[end_imax][1], jmax, MPI_FLOAT, rankR, 0,             // sending to right
                            &p[end_imax+1][1], jmax, MPI_FLOAT, rankR, 0,            // receiving from right
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } /* end of rb */

        /* Partial computation of residual */
        *res = 0.0;
        for (i = initial_imax; i <= end_imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) - 
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    *res += add*add;
                }
            }
        }

        MPI_Allreduce(&*res,&*res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );

        *res = sqrt((*res)/ifull)/p0;
        /* convergence? */
        if (*res<eps) break;
    } /* end of iter */

    MPI_Allgatherv(&p[initial_imax][1], total_imax_iteration_by_rank*(jmax+2), MPI_FLOAT, &p[1][1],counts,displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD );
    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely)
{
    int i, j;
    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int initial_imax,end_imax,total_imax_iteration_by_rank;
    total_imax_iteration_by_rank = floor(imax/size);
    initial_imax = rank*total_imax_iteration_by_rank;
    end_imax = initial_imax + total_imax_iteration_by_rank;
    initial_imax++;

    int displacements[size];
    int counts[size];

    for (int index = 0; index < size; index++)
    {
        displacements[index] = 0;
        counts[index] = total_imax_iteration_by_rank*(jmax+2);
        if(imax%size !=0){
            counts[size-1] = (total_imax_iteration_by_rank + (imax%size))*(jmax+2);
        }
    }

    for (int index = 1; index < size; index++)
    {
        displacements[index] = displacements[index-1] + total_imax_iteration_by_rank*(jmax+2);
    }

    if( (rank == size -1) && (imax%size != 0)){ 
        total_imax_iteration_by_rank += (imax%size);
        end_imax = end_imax + (imax%size);
    }
 
    if (rank == (size-1)){end_imax = end_imax - 1;} 

    for (i=initial_imax; i<=end_imax; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }

    MPI_Allgatherv(&u[initial_imax][1], total_imax_iteration_by_rank*(jmax+2), MPI_FLOAT, &u[1][1], counts,displacements, MPI_FLOAT, MPI_COMM_WORLD);


    if (rank == (size-1)){end_imax = end_imax + 1;} 
    for (i=initial_imax; i<=end_imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
        }
    }

    MPI_Allgatherv(&v[initial_imax][1], total_imax_iteration_by_rank*(jmax+2), MPI_FLOAT, &v[1][1], counts,displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD );
}


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau)
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe; 

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        umax = 1.0e-10;
        vmax = 1.0e-10; 
        for (i=0; i<=imax+1; i++) {
            for (j=1; j<=jmax+1; j++) {
                umax = max(fabs(u[i][j]), umax);
            }
        }
        for (i=1; i<=imax+1; i++) {
            for (j=0; j<=jmax+1; j++) {
                vmax = max(fabs(v[i][j]), vmax);
            }
        }

        deltu = delx/umax;
        deltv = dely/vmax; 
        deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

        if (deltu<deltv) {
            *del_t = min(deltu, deltRe);
        } else {
            *del_t = min(deltv, deltRe);
        }
        *del_t = tau * (*del_t); /* multiply by safety factor */
    }
}
