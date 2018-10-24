#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
// + usare argp?


// number of slices for the potential of the walls
#define M 20
// number of particles:
#define N 32
// frequency of acquisition and storage of thermodinamic variables
#define MEASUREMENT_PERIOD 100


struct Sim sMC(double rho, double T, const double *W, int maxsteps);
void vecBoxMuller(double sigma, size_t length, double *A);
void shiftSystem(double *r, double L);
void initializeWalls(double L, double x0m, double x0sigma, double ym, double ymsigma, double *W);
void initializeBox(double L, int n, double *X);
void markovProbability(const double *X, double *Y, double L, double T, double s, double d, double *ap);
void forces(const double *r, double L, double *F);
void wallsForces(const double *r, const double *W, double L, double *F);
double energy(const double *r, double L);
double pressure(const double *r, double L);
void zeros(size_t length, double *A);
double sum(const double *A, size_t length);
double mean(const double *A, size_t length);
void elforel(const double *A, const double * B, double * C, size_t length);


struct Sim {    // struct containing all the useful results of one simulation
    double E;
    double P;
} sim;


int main(int argc, char** argv)
{
    int maxsteps = 10;
    double rho = 0.1;
    double T = 0.3;
    double L = cbrt(N/rho);
    
    // contains the parameters c and d for every piece of the wall
    double * W = calloc(2*M, sizeof(double));
    
    // parameters of Lennard-Jones potentials of the walls (average and sigma of a gaussian)
    double x0m = 1.0;
    double x0sigma = 0.2;
    double ym = 1.2;
    double ymsigma = 0.3;
    
    initializeWalls(L, x0m, x0sigma, ym, ymsigma, W);
    
    // declaration of the results of the simulations
    struct Sim MC1;
    
    MC1 = sMC(rho, T, W, maxsteps);

    printf("\n%lf\n", MC1.E);
    
    free(W);
    return 0;
}


struct Sim sMC(double rho, double T, const double *W, int maxsteps)   
{
    srand(time(NULL));  // metterne uno per processo MPI
    clock_t start, end;
    
    double sim_time, eta;
    // contains the initial particle positions
    double * X = calloc(3*N, sizeof(double));
    // contains the proposed particle positions
    double * Y = calloc(3*N, sizeof(double));
    double * ap = calloc(3*N, sizeof(double)); // oppure lungo solo N?
    double * E = calloc(maxsteps, sizeof(double));
    double * P = calloc(maxsteps, sizeof(double));
    double * jj = calloc(maxsteps, sizeof(double));

    FILE *positions = fopen("positions.csv", "w");
    for (int n=0; n<N; n++)
        fprintf(positions, "x%d,y%d,z%d,", n+1, n+1, n+1);
    fprintf(positions, "\n");
    
    
    // System initialization
    double D = 2e-3;
    double g = 0.065;
    double L = cbrt(N/rho);
    double a = L/(int)(cbrt(N/4));
    double A = g*T;
    double s = sqrt(4*A*d)/g;
    
    initializeBox(L, N, X); // da sostituire con cavity
    
    
    // Thermalization


    
    // Actual simulation
    start = clock();

    for (int n=0; n<maxsteps; n++)
    {
        E[n] = energy(X, L);
        P[n] = pressure(X, L);
        
        markovProbability(X, Y, L, T, s, D, ap);
        
        for (int i=0; i<3*N; i++)   {
            eta = rand();
            if (eta < ap[i])
            {
                X[i] = Y[i];
                jj[n] += 1;
            }
        }

        // mettere in ciclo for separato?
        printf("%f\t%f\t%f\n", E[n], P[n], jj[n]);
        for (int i=0; i<3*N; i++)
            fprintf(positions, "%0.18lf,", X[i]);

        fprintf(positions, "\n");

    }
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime: %lf s\n\n", sim_time);

    
    // Opens csv file where it then writes a table with the data
    FILE *data = fopen("data.csv", "w");
    if (data == NULL || positions == NULL)
        perror("error while writing on data.csv");

    for (int i=0; i<maxsteps; i++)
        fprintf(data, "%0.18lf,%0.18lf,%0.18lf\n", E[i], P[i], jj[i]);

    // Create struct of the mean values and deviations
    struct Sim results;
    results.E = 42.0;
    results.P = mean(P, maxsteps);

    // frees the allocated memory
    free(X); free(Y); free(E); free(P); free(jj); free(ap);

    return results;
}


void markovProbability(const double *X, double *Y, double L, double T, double s, double D, double *ap)  
{
    double * gauss = malloc(3*N * sizeof(double));
    double * FX = calloc(3*N, sizeof(double));
    double * FY = calloc(3*N, sizeof(double));
    double * displacement = malloc(3*N * sizeof(double));
    double * WX = malloc(3*N * sizeof(double));
    double * WY = malloc(3*N * sizeof(double));
    
    vecBoxMuller(s, 3*N, gauss);
    forces(X, L, FX);
    
    // Proposal
    for (int i=0; i<3*N; i++)
    {
        Y[i] = X[i] + D*FX[i] + gauss[i]*s;     // force da dividere per γ?
        displacement[i] = Y[i] - X[i];   // usare shiftSystem() anche su questo?
    }
    
    shiftSystem(Y, L);
    forces(Y, L, FY);
    double Uxy = energy(X,L) - energy(Y,L);
    
    // Acceptance probability calculation
    for (int i=0; i<3*N; i++)     // DA SISTEMARE
    {
        WX[i] = (displacement[i] - FX[i]*D) * (displacement[i] - FX[i]*D);  // controllare segni
        WY[i] = (- displacement[i] - FY[i]*D) * (- displacement[i] - FY[i]*D);
        
        ap[i] = exp(Uxy/T + (WX[i]-WY[i])/(4*D*T));
    }
    
    free(gauss); free(FX); free(FY); free(displacement); free(WX); free(WY); 
}

void initializeBox(double L, int N_, double *X) 
{
    int Na = (int)(cbrt(N_/4)); // number of cells per dimension
    double a = L / Na;  // interparticle distance
    if (Na != cbrt(N_/4))
        perror("Can't make a cubic FCC crystal with this N :(");


    for (int i=0; i<Na; i++)    {   // loop over every cell of the fcc lattice
        for (int j=0; j<Na; j++)    {
            for (int k=0; k<Na; k++)    {
                int n = i*Na*Na + j*Na + k; // unique number for each triplet i,j,k
                X[n*12+0] = a*i;
                X[n*12+1] = a*j;
                X[n*12+2] = a*k;

                X[n*12+3] = a*i + a/2;
                X[n*12+4] = a*j + a/2;
                X[n*12+5] = a*k;

                X[n*12+6] = a*i + a/2;
                X[n*12+7] = a*j;
                X[n*12+8] = a*k + a/2;

                X[n*12+9] = a*i;
                X[n*12+10] = a*j + a/2;
                X[n*12+11] = a*k + a/2;
            }
        }
    }

    for (int n=0; n<3*N; n++)
        X[n] += a/4;   // needed to avoid particles exactly at the edges of the box

    shiftSystem(X,L);
}


// TODO
void initializeCavity(double L, int N_, double *X) // da rendere rettangolare?
{
    int Na = (int)(cbrt(N_/4)); // number of cells per dimension
    double a = L / Na;  // interparticle distance
    if (Na != cbrt(N_/4))
        perror("Can't make a cubic FCC crystal with this N :(");


    for (int i=0; i<Na; i++)    {   // loop over every cell of the fcc lattice
        for (int j=0; j<Na; j++)    {
            for (int k=0; k<Na; k++)    {
                int n = i*Na*Na + j*Na + k; // unique number for each triplet i,j,k
                X[n*12+0] = a*i;
                X[n*12+1] = a*j;
                X[n*12+2] = a*k;

                X[n*12+3] = a*i + a/2;
                X[n*12+4] = a*j + a/2;
                X[n*12+5] = a*k;

                X[n*12+6] = a*i + a/2;
                X[n*12+7] = a*j;
                X[n*12+8] = a*k + a/2;

                X[n*12+9] = a*i;
                X[n*12+10] = a*j + a/2;
                X[n*12+11] = a*k + a/2;
            }
        }
    }

    for (int n=0; n<3*N; n++)
        X[n] += a/4;   // needed to avoid particles exactly at the edges of the box

    shiftSystem(X,L);
}


/*
 Takes average and standard deviation of the distance from the y-axis and of the maximum binding energy
 and puts in the W array two gaussian distributions of those parameters
*/
// Per ora divide la parete in M fettine rettangolari

void initializeWalls(double L, double x0m, double x0sigma, double ym, double ymsigma, double *W)    
{
    double * X0 = malloc(M * sizeof(double));
    double * YM = malloc(M * sizeof(double));
    
    vecBoxMuller(x0sigma, M, X0);
    vecBoxMuller(ymsigma, M, YM);
    
    for (int l=0; l<M; l++)  {
        W[2*l] = 2*pow(X0[l]+x0m, 12.) * (YM[l]+ym)*(YM[l]+ym);
        W[2*l+1] = pow(X0[l]+x0m, 6.) * (YM[l]+ym);
    }
    
    free(X0); free(YM);
}


void forces(const double *r, double L, double *F) 
{
    double dx, dy, dz, dr2, dV;

    for (int l=0; l<N; l++)  {
         for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)    {
                //dV = -der_LJ(sqrt(dr2))
                dV = -24*(1.0/(dr2*dr2*dr2*dr2)) + 48*1.0/pow(dr2,7.0);
                F[3*l+0] += dV*dx;
                F[3*l+1] += dV*dy;
                F[3*l+2] += dV*dz;
                F[3*i+0] -= dV*dx;
                F[3*i+1] -= dV*dy;
                F[3*i+2] -= dV*dz;
            }
        }
    }
}


/*
 * Calculate the force exerted on the particles by the surface.
 * Be careful, it doesn't initialize F to zero, it only adds.
 * 
*/

// TODO
// sentire forza da tutti i segmenti? Solo quelli più vicini?

void wallsForces(const double *r, const double *W, double L, double *F) 
{ 
    double dx, dy, dz, dr2, dV;
     
    for (int n=0; n<N; n++)  {
            if (dr2 < L*L/4)    {
                //dV = -der_LJ(sqrt(dr2))
                dV = -24 * W[1] / (dr2*dr2*dr2*dr2) + 48 * W[2] / pow(dr2,7.0);
                F[3*n+0] += dV*dx;
                F[3*n+1] += dV*dy;
                F[3*n+2] += dV*dz;
            }
    }
}


/*
 * Calculate the interparticle potential energy of the system
 * 
*/

double energy(const double *r, double L)  
{
    double V = 0.0;
    double dx, dy, dz, dr2;
    for (int l=0; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                V += 1.0/(dr2*dr2*dr2*dr2*dr2*dr2) - 1.0/(dr2*dr2*dr2);
        }
    }
    return V*4;
}

/*
 * Calculate the potential energy between the particles and the wall
 * 
*/

double wallsEnergy(const double *r, double *W, double L)  
{
    double V = 0.0;
    double dz2;
    for (int n=0; n<N; n++)  {
            dz2 = r[3*n+2] * r[3*n+2];
            V += W[1] / pow(pow(dz2,3.),2.) - W[2] / (dz2*dz2*dz2);
    }
    return V*4;
}



double pressure(const double *r, double L)  
{
    double P = 0.0;
    double dx, dy, dz, dr2;
    for (int l=0; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                P += 4*(6*pow(dr2,-3.) - 12*pow(dr2,-6.));   // ricontrollare
        }
    }
    return -P/(3*L*L*L);
}


inline void shiftSystem(double *r, double L)  // da ricontrollare
{
    for (int j=0; j<3*N; j++)
        r[j] = r[j] - L*rint(r[j]/L);
}

inline void shiftSystem2D(double *r, double L)  // da ricontrollare
{
    for (int j=0; j<N; j++) {
        r[3*j] = r[3*j] - L*rint(r[3*j]/L);
        r[3*j+1] = r[3*j+1] - L*rint(r[3*j+1]/L);
    }
}


/*
 * Put in the array A gaussian-distributed numbers around 0, with standard deviation sigma
 * // confrontare con dSFMT
*/

inline void vecBoxMuller(double sigma, size_t length, double * A)
{
    double x1, x2;

    for (int i=0; i<round(length/2); i++) {
        x1 = (double)rand() / (double)RAND_MAX;
        x2 = (double)rand() / (double)RAND_MAX;
        A[2*i] = sqrt(-2*sigma*log(1-x1))*cos(2*M_PI*x2);
        A[2*i+1] = sqrt(-2*sigma*log(1-x2))*sin(2*M_PI*x1);
    }
}


/*
 * Calculate the autocorrelation function
 * 
*/

void fft_acf(const double *H, int k_max, size_t length, double * acf)   
{
    fftw_plan p;
    int lfft = length/2+1;
    double * Z = fftw_malloc(length * sizeof(double));
    complex double * fvi = fftw_malloc(lfft * sizeof(fftw_complex));
    double * temp = fftw_malloc(length * sizeof(double));
    complex double * C_H = fftw_malloc(lfft * sizeof(fftw_complex));

    double meanH = mean(H, length);
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;

    p = fftw_plan_dft_r2c_1d(lfft, Z, fvi, FFTW_FORWARD);
    fftw_execute(p);

    for (int i=0; i<lfft; i++)  // compute the abs2 of the transform
        temp[i] = fvi[i] * conj(fvi[i]);

    p = fftw_plan_dft_r2c_1d(lfft, temp, C_H, FFTW_BACKWARD);
    fftw_execute(p);

    for (int i=0; i<k_max; i++)
        acf[i] = creal(C_H[i]) / creal(C_H[1]);


    fftw_destroy_plan(p);
    fftw_free(fvi); fftw_free(C_H); free(temp); free(Z);
}





/*
    Mathematics rubbish
*/

inline double sum(const double * A, size_t length)   {
    double s = 0.;
    for (int i=0; i<length; i++)
        s += A[i];

    return s;
}

inline double dot(const double * A, double * B, size_t length)  {
    double result = 0.0;

    for (int i=0; i<length; i++)
        result += A[i]*B[i];

    return result;
}

inline void elforel(const double * A, const double * B, double * C, size_t length)  {
    for (int i=0; i<length; i++)
        C[i] = A[i]*B[i];
}

inline double mean(const double * A, size_t length) {
    return sum(A,length)/length;
}

double media(const double * A, size_t length) {
    double birra = 0.4;
    return birra;
}

inline void zeros(size_t length, double *A)
{
    for (int i=length; i!=0; i--)
        A[i] = 0.0;
}

inline double variance(const double * A, size_t length)  {
    double * A2 = malloc(length * sizeof(double));
    elforel(A,A,A2,length);
    double var = mean(A2,length) - mean(A,length)*mean(A,length);
    free(A2);
    return var;
}


/* TODO
 * 
 * mettere funzioni in file.h esterno
 * 
 * Prestazioni: 
 *
 * provare N come #define
 * FX, FY, WX, WY preallocati?
 * dSMT al post di rand()
 * sostituire funzioni stupide con macro (probabilmente inutile - compilatore + inlining)
 * confrontare prestazioni con array in stack
 * provare loop-jamming
 * provare loop al contrario  (probabilmente inutile - compilatore)
 * provare uint_fast8_t (probabilmente inutile)
 * aliases per funzioni dentro funzioni con array in lettura
 * bitwise shift per moltiplicare/dividere per 2  (probabilmente inutile - compilatore)
 * provare c++ con vectorizer di Agner
 * 
 */
