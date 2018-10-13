#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>


double * initializeSystem(double *X, double L, int N);
void shiftSystem(double *r, double L, int N);
double energy(double *r, double L, int N);
double pressure(double *r, double L, int N);


int main(int argc, char** argv)
{
    srand(time(NULL));  // metterne uno per processo MPI
    const int N = 32;
    const int maxsteps = 10^5;
    double * X = malloc(3*N * sizeof(double));
    double * Y = malloc(3*N * sizeof(double));
    double * E = malloc(maxsteps * sizeof(double));
    double * P = malloc(maxsteps * sizeof(double));
    double * jj = malloc(maxsteps * sizeof(double));
    
    double rho = 0.1;
    double L = cbrt(N/rho);
    double a = L/(int)(cbrt(N/4));
    
    X = initializeSystem(X, L, N);
    
    for (int i=0; i<maxsteps; i++)
    {
        
    }
    
    
    // Opens csv file where it then writes a table with the data
    FILE *f = fopen("phase_space.csv", "w");
    if (f == NULL) return -1;
    
    for (int i=0; i<maxsteps; i++)
    {
        printf("%f\t%f\t%f\n", E[i], P[i], jj[i]);
        fprintf(f, "%0.18lf,%0.18lf,%0.18lf\n", E[i], P[i], jj[i]);    // writes in the .csv file
    }
    
    // frees the allocated memory
    free(X); free(Y); free(E); free(P); free(jj);
    return 0;
}

double * initializeSystem(double *X, double L, int N) {
    int Na = (int)(cbrt(N/4)); // number of cells per dimension
    double a = L / Na;  // passo reticolare
    (Na != cbrt(N/4)) && printf("Can't make a cubic FCC crystal with this N :(");

    
    for (int i=0; i<Na; i++)    {   // loop over every cell of the fcc lattice
        for (int j=0; j<Na; j++)    {   
            for (int k=0; k<Na; k++)    {   
                int n = i*Na*Na + j*Na + k; // unique number for each triplet i,j,k
                X[n*12+1] = a*i;
                X[n*12+2] = a*j;
                X[n*12+3] = a*k;
                
                X[n*12+4] = a*i + a/2;
                X[n*12+5] = a*j + a/2;
                X[n*12+6] = a*k;
                
                X[n*12+7] = a*i + a/2;
                X[n*12+8] = a*j;
                X[n*12+9] = a*k + a/2;
                
                X[n*12+10] = a*i;
                X[n*12+11] = a*j + a/2;
                X[n*12+12] = a*k + a/2;
            }
        }
    }
    
    for (int n=0; n<N; n++) 
        X[n] += a/4;   // needed to avoid particles exactly at the edges of the box
    
    shiftSystem(X,L,N);
    return X;
}


void shiftSystem(double *r, double L, int N)  {  // da sistemare
    for (int j=0; j<3*N; j++)
        r[j] = r[j] - L*round(r[j]/L);
}


double energy(double *r, double L, int N)  {
    double V = 0.0;
    double dx, dy, dz, dr2;
    for (int l=0; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l+1] - r[3*i+1];
            dx = dx - L*round(dx/L);
            dy = r[3*l+2] - r[3*i+2];
            dy = dy - L*round(dy/L);
            dz = r[3*l+3] - r[3*i+3];
            dz = dz - L*round(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                V += 4*(1.0/pow(pow(dr2,3.),2.) - 1.0/(dr2*dr2*dr2));
        }
    }
    return V;
}

double pressure(double *r, double L, int N)  {
    double P = 0.0;
    double dx, dy, dz, dr2;
    for (int l=0; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l+1] - r[3*i+1];
            dx = dx - L*round(dx/L);
            dy = r[3*l+2] - r[3*i+2];
            dy = dy - L*round(dy/L);
            dz = r[3*l+3] - r[3*i+3];
            dz = dz - L*round(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                P += 4*(6*pow(dr2,-3.) - 12*pow(dr2,-6.));   // ricontrollare
        }
    }
    return -P/(3*L*L*L);
}

void vecboxMuller(double sigma, size_t N, double * A)  {   // confrontare con dSFMT
    double x1, x2;
    
    for (int i=0; i<round(N/2); i++) {
        x1 = rand()/RAND_MAX;
        x2 = rand()/RAND_MAX;
        A[2*i] = sqrt(-2*sigma*log(1-x1))*cos(2*M_PI*x2);
        A[2*i+1] = sqrt(-2*sigma*log(1-x2))*sin(2*M_PI*x1);
    }
}

void fft_acf(double *H, int k_max, size_t length, double * acf)   {
    
    fftw_plan p;
    int lftt = length/2+1
    double * Z = fftw_malloc(length * sizeof(double));
    double * fvi = fftw_malloc(lfft * sizeof(fftw_complex));
    double * gnam = fftw_malloc(length * sizeof(double));
    double * C_H = fftw_malloc(lfft * sizeof(fftw_complex));
    
    double meanH = mean(H);
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;
    
    p = fftw_plan_dft_r2c_1d(lfft, Z, fvi, FFTW_FORWARD);
    fftw_execute(p);
    
    elforelcpx(fvi, conj(fvi), gnam);  // compute the abs2 of the transform
    
    p = fftw_plan_dft_r2c_1d(lfft, gnam, C_H, FFTW_BACKWARD);
    fftw_execute(p);
    
    for (int i=0; i<k_max; i++)
        acf[i] = creal(C_H[i]) / creal(C_H[1]);

    
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    return acf
}


double dot(double * A, double * B, size_t length)  {
    double result = 0.0;
    
    for (int i=0; i<length; i++)
        result += A[i]*B[i];
        
    return result;
}
void elforel(double * A, double * B, double * C, size_t length)  {  
    for (int i=0; i<length; i++)
        C[i] = A[i]*B[i];
}
double mean(double * A, size_t length) {
    double result = 0.0;
    for (int i=0; i<length; i++)
        result += A[i];
    
    return result/length;
}
double media(double * A, size_t length) {
    double birra = 0.4;
    return birra;
}
double variance(double * A, size_t length)  {
    double * A2 = malloc(length * sizeof(double));
    elforel(A,A,A2,length);
    double var = mean(A2,length) - mean(A,length)*mean(A,length);
    free(A2);
    return var;
}

