
/* TODO
 * 
 * Unire calcolo energia e forze interparticellari e con parete?
 * chiedere a utente parametri simulazione
 * deltaX gaussiano sferico o in ogni direzione?
 * decidere cosa mettere come macro e cosa come variabile passata a simulation (tipo gather_lapse)
 * energia totale come 1/2 somma delle energie singole?
 * fare grafico di local density in funzione di E legame
 * passare anche rank processo e metterlo in tutti i printf
 * in generale fare in modo che noMPI venga trattato come se avesse rank 0
 * IMPORTANTE: capire se il L*L/4 può essere lasciato com'è
 * vedere se si può semplificare calcolo distanze da pareti
 * localDensity con numero minore di divisioni lungo z
 * 
 */


#include "SMC.h"


struct Sim sMC(double L, double Lz, double T, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps)   
{
    // System properties
    double rho = N / (L*L*Lz);
    //double gamma = 0.5;
    //double dT = 2e-2;
    //double A = gamma*dT;
    //double s = sqrt(4*A*D)/dT;
    double A = 0.2;
    
    // Data-harvesting parameters
    int gather_steps = (int)(maxsteps/gather_lapse);
    int kmax = 42000;
    int Nv = 30*30*30; // number of cubes dividing the volume, to compute the local density (should be a perfect cube)

    
    clock_t start, end;
    double sim_time;
    srand(time(NULL)); //should be different for each process
    
    printf("Starting new run with %d particles, ", N);
    printf("T=%0.2f, rho=%0.4f, A=%0.3f, for %d steps...\n", T, rho, A, maxsteps);

    
    //copy the initial positions R0 (common to all the simulations) to the local array R
    double *R = malloc(3*N * sizeof(double));
    memcpy(R, R0, 3*N * sizeof(double));
    double * Rn = calloc(3*N, sizeof(double)); // contains the proposed particle positions
    double * E = calloc(maxsteps, sizeof(double));
    double * P = calloc(gather_steps, sizeof(double));
    int * jj = calloc(maxsteps, sizeof(int)); // usare solo in termalizzazione?
    unsigned long * lD = calloc(Nv, sizeof(unsigned long));
    double * acf = calloc(kmax, sizeof(double));    // autocorrelation function
    //double * acf2 = calloc(kmax, sizeof(double));   // da eliminare quando sarò sicuro che fft_acf funziona bene

    
    // Initialize csv files
    char filename[64]; 
    snprintf(filename, 64, "./positions_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    FILE *positions = fopen(filename, "w");
    if (positions == NULL)
        perror("error while writing on positions.csv");
    
    for (int n=0; n<N; n++)
        fprintf(positions, "x%d,y%d,z%d,", n+1, n+1, n+1);
    fprintf(positions, "\n");
    
    snprintf(filename, 64, "./data_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    FILE *data = fopen(filename, "w");
    if (data == NULL)
        perror("error while writing on data.csv");
    
    fprintf(data, "E, P, jj\n");
    
    FILE * localdensity;    // per ora restituisce numero cumulativo, si potrebbe anche fare come con positions
    snprintf(filename, 64, "./localdensity_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    localdensity = fopen(filename, "w");
    if (localdensity == NULL)
        perror("error while writing on localdensity.csv");
    
    fprintf(localdensity, "nx, ny, nz, n\n");
    
    FILE * autocorrelation;
    snprintf(filename, 64, "./autocorrelation_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    autocorrelation = fopen(filename, "w");
    if (autocorrelation == NULL)
        perror("error while writing on autocorrelation.csv");
    
    fprintf(autocorrelation, "CH\n");
    
    
    /*  Thermalization   */

    start = clock();
    for (int n=0; n<eqsteps; n++)
    {
        E[n] = energy(R, L);
        oneParticleMoves(R, Rn, W, L, Lz, A, T, &jj[n]);
    }
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("\nThermalization completed in %f s with", sim_time);
    printf("average acceptance ratio %f, mean energy %f.\n", intmean(jj,eqsteps)/N, mean(E,eqsteps)+3*N*T/2);
    
    for (int n=0; n<eqsteps; n++)
        jj[n] = 0;
    
    
    
    /*  Actual simulation   */
    
    printf("The expected time of execution is ~%0.1f minutes.\n", 1.03*sim_time*maxsteps/eqsteps/60);
    start = clock();

    for (int n=0; n<maxsteps; n++)
    {
        if (n % gather_lapse == 0)  {
            int k = (int)(n/gather_lapse);
            
            P[k] = pressure(R, L, Lz);
            //P[k] += wallsPressure(R, W, L, Lz);
            localDensity(R, L, Lz, Nv, lD); // add the number of particles in each block of the volume
            
            for (int i=0; i<3*N; i++)
                fprintf(positions, "%0.12lf,", R[i]);

            fprintf(positions, "\n");
        }
        
        E[n] = energy(R, L);  // da calcolare in modo più intelligente dentro oneParticleMoves
        E[n] += wallsEnergy(R, W, L, Lz);
        
        oneParticleMoves(R, Rn, W, L, Lz, A, T, &jj[n]);
    }
    
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime: %f s\n", sim_time);

    
    
    /*  Data preparation and storage  */

    // total energy and total pressure
    for (int k=0; k<gather_steps; k++)
        P[k] += rho*T;
    
    for (int n=0; n<maxsteps; n++)
        E[n] += 3*N*T/2;
    
    // save temporal data of the system (gather_steps arrays of energy, pressure and acceptance ratio)
    for (int k=0; k<gather_steps; k++)
        fprintf(data, "%0.9lf, %0.9lf, %d\n", E[k*gather_lapse], P[k]+rho*T, jj[k]);
    
    int Nl = (int) rint(cbrt(Nv)); // number of cells per dimension
    for (int i=0; i<Nl; i++)    {
        for (int j=0; j<Nl; j++)    {
            for (int k=0; k<Nl; k++)    {
                int v = i*Nl*Nl + j*Nl + k;
                fprintf(localdensity, "%d, %d, %d, %lu\n", i, j, k, lD[v]);
            }
        }
    }
    

    // autocorrelation calculation
    fft_acf(E, maxsteps, kmax, acf);
    double tau = sum(acf,kmax);
    //simple_acf(E, maxsteps, kmax, acf2);    // da eliminare dopo aver confrontato
    //printf("TauSimple: %f \n", tau);
    
    for (int m=0; m<kmax; m++)
      fprintf(autocorrelation, "%0.6lf\n", acf[m]);
    

    // Create struct of the mean values and deviations to return
    struct Sim results;
    results.E = mean(E, maxsteps);
    results.dE = sqrt(variance_corr(E, tau, maxsteps));
    results.P = mean(P, gather_steps);
    results.dP = sqrt(variance(P, gather_steps));
    results.acceptance_ratio = intmean(jj, maxsteps)/N;
    results.tau = tau;// * gather_lapse;
    results.cv = variance_corr(E, tau, maxsteps) / (T*T);
    printf("tau_noncorr = %f \n", variance(E, maxsteps) / (T*T));
    
    memcpy(results.Rfinal, R, 3*N * sizeof(double));
    
    struct DoubleArray ACF; // capire come allocare la memoria nel modo giusto
    ACF.length = kmax;
    ACF.data = acf;
    results.ACF = ACF;

    // free the allocated memory
    free(R); free(Rn); free(E); free(P); free(jj); free(acf); free(lD);
    fclose(positions); fclose(data); fclose(autocorrelation); fclose(localdensity);

    return results;
}



/*
 * Executes a single particle Smart Monte Carlo step for each of the N particles.
 * Each times it starts the loop from a random particle (not sure if this is useful...)
 * It modifies the passed arrays R, Rn and j (the last one containing the ratio of accepted steps).
 * 
*/

void oneParticleMoves(double * R, double * Rn, const double * W, double L, double Lz, double A, double T, int * j)
{
    double * displ = malloc(3*N * sizeof(double));
    double Um, Un, deltaX, deltaY, deltaZ, Fmx, Fmy, Fmz, Fnx, Fny, Fnz, deltaW, ap;
        
    vecBoxMuller(sqrt(2*A), 3*N, displ);
    
    for (int i=0; i<3*N; i++)   // controllare se è necessario qua o si può usare solo una volta all'inizio
        Rn[i] = R[i];
    
    // at each oneParicleMoves call, we start moving a different particle (offset % N)
    int n; int offset = rand();
    
    for (int nn=0; nn<N; nn++)
    {
        n = (nn+offset)%N;

        // calculate the potential energy of particle n, first due to other particles and then to the wall
        Um = energySingle(R, L, n);
        Um += wallsEnergySingle(R[3*n], R[3*n+1], R[3*n+2], W, L, Lz);
        
        // same thing for the force exerted on particle n
        forceSingle(R, L, n, &Fmx, &Fmy, &Fmz);
        wallsForce(R[3*n], R[3*n+1], R[3*n+2], W, L, Lz, &Fmx, &Fmy, &Fmz);

        // calculate the proposed new position of particle n
        deltaX = Fmx*A/T + displ[3*n];
        deltaY = Fmy*A/T + displ[3*n+1];
        deltaZ = Fmz*A/T + displ[3*n+2];
        Rn[3*n] = R[3*n] + deltaX;
        Rn[3*n+1] = R[3*n+1] + deltaY;
        Rn[3*n+2] = R[3*n+2] + deltaZ;
        
        shiftSystem3D(Rn, L, Lz);   // verificare che vada bene qui, e che vada messo il 2D e non il 3D

        // calculate energy and forces in the proposed new position
        Un = energySingle(Rn, L, n);
        Un += wallsEnergySingle(Rn[3*n], Rn[3*n+1], Rn[3*n+2], W, L, Lz);
        forceSingle(Rn, L, n, &Fnx, &Fny, &Fnz);
        wallsForce(Rn[3*n], Rn[3*n+1], Rn[3*n+2], W, L, Lz, &Fnx, &Fny, &Fnz);


        // Calculate the acceptance probability for the single-particle move
        
        deltaW = ((Fnx-Fmx)*(Fnx-Fmx) + (Fny-Fmy)*(Fny-Fmy) + (Fnz-Fmz)*(Fnz-Fmz) +
            2*((Fnx-Fmx)*Fmx + (Fny-Fmy)*Fmy + (Fnz-Fmz)*Fmz)) * A/(4*T);

        ap = exp(-(Un-Um + (deltaX*(Fnx+Fmx) + deltaY*(Fny+Fmy) + deltaZ*(Fnz+Fmz))/2 + deltaW)/T);

        
        // Accepts the move by comparing ap with a random uniformly distributed probability
        // If the move is rejected, return Rn to the initial state (equal to R)
        
        if ((double)rand()/(double)RAND_MAX < ap)
        {
            R[3*n] = Rn[3*n];
            R[3*n+1] = Rn[3*n+1];
            R[3*n+2] = Rn[3*n+2];
            *j += 1;
        }
        else    {
            Rn[3*n] = R[3*n];
            Rn[3*n+1] = R[3*n+1];
            Rn[3*n+2] = R[3*n+2];
        }
    }

    free(displ);
}


/*
 * Old SMC step, doesn't work ¯\_(ツ)_/¯
 * 
*/
/*
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
    double Uxy = energy(Y,L) - energy(X,L); // da cambiare in modo da non dover ricalcolare l'energia salvata
    
    // Acceptance probability calculation
    for (int i=0; i<3*N; i++)     // DA SISTEMARE
    {
        WX[i] = (displacement[i] - FX[i]*D) * (displacement[i] - FX[i]*D);  // controllare segni
        WY[i] = (- displacement[i] - FY[i]*D) * (- displacement[i] - FY[i]*D);
        
        ap[i] = exp(-Uxy/T + (WX[i]-WY[i])/(4*D*T));
    }
    
    free(gauss); free(FX); free(FY); free(displacement); free(WX); free(WY); 
}
*/



/* ###############    System preparation and other useful routines    ############### */


/*
 * Initialize the particle positions X as a fcc crystal centered around (0, 0, 0)
 * with the shape of a cube with L = cbrt(V)
*/ //TODO vedere se è meglio usare az e allungare reticolo

void initializeBox(double L, double Lz, int N_, double *X) 
{
    int Na = (int)(cbrt(N_/4)); // number of cells per dimension
    double a = L / Na;
    double az = Lz / Na;
    if ( !isApproxEqual((double) Na, cbrt(N_/4)) )
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

    shiftSystem3D(X,L,Lz);
}


/*
 Takes average and standard deviation of the distance from the y-axis (at f(x)=0) and of the maximum binding energy
 and puts in the W array two gaussian distributions of resulting parameters "a" and "b".
 These parameters enter the Lennard-Jones potential in the form V = 4*(a/r^12 - b/r^6).
*/
// Per ora divide la parete in M fettine rettangolari, e il potenziale verrà generato da una linea per ogni fettina 

void initializeWalls(double x0m, double x0sigma, double ymm, double ymsigma, double *W, FILE * wall)    
{
    double * X0 = malloc(M*M * sizeof(double));
    double * YM = malloc(M*M * sizeof(double));
    
    vecBoxMuller(x0sigma, M*M, X0);
    vecBoxMuller(ymsigma, M*M, YM);
    
    // saves the parameter distribution to a file (the rest of the program only uses a and b instead)
    fprintf(wall, "nx, ny, x0, ymin\n");
    
    for (int i=0; i<M; i++) {
        for (int j=0; j<M; j++) {
            int m = j + i*M;
            fprintf(wall, "%d, %d, %f, %f\n", i, j, X0[m]+x0m, YM[m]+ymm);
            W[2*m] = pow(X0[m]+x0m, 12.) * (YM[m]+ymm)*(YM[m]+ymm);     // a
            W[2*m+1] = pow(X0[m]+x0m, 6.) * (YM[m]+ymm);                // b
        }
    }
    
    
    free(X0); free(YM);
}


/*
 * Put in the array A gaussian-distributed numbers around 0, with standard deviation sigma
 * 
*/

inline void vecBoxMuller(double sigma, size_t length, double * A)
{
    double x1, x2;

    for (int i=0; i<round(length/2); i++) {
        x1 = (double) rand() / (RAND_MAX + 1.0);
        x2 = (double) rand() / (RAND_MAX + 1.0);
        A[2*i] = sigma * sqrt(-2*log(1-x1)) * cos(2*M_PI*x2);
        A[2*i+1] = sigma * sqrt(-2*log(1-x2)) * sin(2*M_PI*x1);
    }
}


inline void shiftSystem(double *r, double L)
{
    for (int j=0; j<3*N; j++)   {
        r[j] = r[j] - L*rint(r[j]/L);
    }
}

inline void shiftSystem3D(double *r, double L, double Lz)
{
    for (int j=0; j<N; j++) {
        r[3*j] = r[3*j] - L*rint(r[3*j]/L);
        r[3*j+1] = r[3*j+1] - L*rint(r[3*j+1]/L);
        r[3*j+2] = r[3*j+2] - Lz*rint(r[3*j+2]/Lz);
    }
}

// TODO chiedere se va bene tralasciare lungo z
inline void shiftSystem2D(double *r, double L)
{
    for (int j=0; j<N; j++) {
        r[3*j] = r[3*j] - L*rint(r[3*j]/L);
        r[3*j+1] = r[3*j+1] - L*rint(r[3*j+1]/L);
    }
}






/* ###############    Physical quantities of interest or needed for the system evolution    ############### */


/*
 * Calculate the potential energy of particle i with respect to other particles
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4
*/

double energySingle(const double *r, double L, int i)
{
    double V = 0.0;
    double dx, dy, dz, dr2, dr6;
    for (int l=0; l<N; l++)  
    {
        if (l != i)   
        {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            //dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            
            if (dr2 < L*L/4)
            {
                dr6 = dr2*dr2*dr2;
                V += 1.0/(dr6*dr6) - 1.0/dr6;
            }
        }
    }
    return V*4;
}


/*
 * Calculate the forces acting on particle i due to the interaction with other particles
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4 (?)
*/

void forceSingle(const double *r, double L, int i, double *Fx, double *Fy, double *Fz) 
{
    double dx, dy, dz, dr2, dr8, dV;
    *Fx = 0.0;
    *Fy = 0.0;
    *Fz = 0.0;

    for (int l=0; l<N; l++)  
    {
         if (l != i)   
         {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            //dz = dz - Lz*rint(dz/Lz); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
            {
                dr8 = dr2*dr2*dr2*dr2;
                dV = 24.0/dr8 - 48.0/(dr8*dr2*dr2*dr2);
                *Fx -= dV*dx;
                *Fy -= dV*dy;
                *Fz -= dV*dz;
            }
        }
    }
}



/*
 * Calculate the interparticle potential energy of the system
*/

double energy(const double *r, double L)  
{
    double V = 0.0;
    double dx, dy, dz, dr2;
    for (int l=1; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            //dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                V += 1.0/(dr2*dr2*dr2*dr2*dr2*dr2) - 1.0/(dr2*dr2*dr2);
        }
    }
    return V*4;
}



/*
 * Calculate the 3N array of forces acting on each particles due to the interaction with other particles
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4 (?)
*/
// Manca funzione analoga per contributo pareti (servirà?)

void forces(const double *r, double L, double *F)
{
    double dx, dy, dz, dr2, dr8, dV;

    for (int l=1; l<N; l++)  
    {
         for (int i=0; i<l; i++)   
         {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            //dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)    
            {
                //dV = der_LJ(sqrt(dr2))
                dr8 = dr2*dr2*dr2*dr2;
                dV = 24.0/dr8 - 48.0/(dr8*dr2*dr2*dr2);
                F[3*l+0] -= dV*dx;
                F[3*l+1] -= dV*dy;
                F[3*l+2] -= dV*dz;
                F[3*i+0] += dV*dx;
                F[3*i+1] += dV*dy;
                F[3*i+2] += dV*dz;
            }
        }
    }
}



/*
 * Calculates ONLY the pressure due to the virial contribution,
 * but from both particle-particle and wall-particle interactions.
 * 
*/

double pressure(const double *r, double L, double Lz)  
{
    double P = 0.0;
    double dx, dy, dz, dr2, dr6;
    for (int l=1; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            //dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)    
            {
                //P += 24*pow(dr2,-3.) - 48*pow(dr2,-6.);
                dr6 = dr2*dr2*dr2;
                P += 24.0/dr6 - 48.0/(dr6*dr6);
            }
        }
    }
    return -P/(3*L*L*Lz);
}



/*
 * Calculate the potential energy between one particle and the wall
 * 
*/

double wallsEnergySingle(double rx, double ry, double rz, const double * W, double L, double Lz)
{
    double V = 0.0;
    double dx, dy, dz, dr2, dr6;
    double dw = L/M;    // distance between two wall elements
    
    for (int i=0; i<M; i++) {
        for (int j=0; j<M; j++) {
            int m = j + i*M;
            dx = rx - i*dw - dw/2;
            dx = dx - L*rint(dx/L);
            dy = ry - j*dw - dw/2;
            dy = dy - L*rint(dy/L);
            dz = rz + Lz/2;
            dz = dz - Lz*rint(dz/Lz);
            dr2 = dx*dx + dy*dy + dz*dz;
        
            if (dr2 < L*L/4)
            {
                dr6 = dr2*dr2*dr2;
                V += W[2*m]/(dr6*dr6) - W[2*m+1]/dr6;
            }
        }
    }
    return V*4;
}



/*
 * Calculate the force exerted on one particle by the surface.
 * Be careful, it doesn't initialize F to zero, it only adds.
 * 
*/
// sentire forza da tutti i segmenti? Solo quelli più vicini?

void wallsForce(double rx, double ry, double rz, const double * W, double L, double Lz, double *Fx, double *Fy, double *Fz) 
{ 
    double dx, dy, dz, dr2, dr8, dV;
    double dw = L/M;    // distance between consecutive wall potential sources
    
   for (int i=0; i<M; i++) 
   {
        for (int j=0; j<M; j++) 
        {
            int m = j + i*M;
            dx = rx - i*dw - dw/2;
            dx = dx - L*rint(dx/L);
            dy = ry - j*dw - dw/2;
            dy = dy - L*rint(dy/L);
            // se rz è positivo, rint dà 1 e la distanza è calcolata da parete sopra. 
            // Infatti dz = (rz-L/2) < 0, forza "in direzione" delle z negative
            // se rz è negativo, dz = (rz+L/2) > 0
            // TODO controllare segno e/o casi in cui potrebbe dare risultati non voluti
            dz = rz + Lz/2;
            dz = dz - Lz*rint(dz/Lz);
            dr2 = dx*dx + dy*dy + dz*dz;
        
            if (dr2 < L*L/4)
            {
                dr8 = dr2*dr2*dr2*dr2;
                dV = 24.0 * W[2*m+1] / dr8 - 48.0 * W[2*m] / (dr8*dr2*dr2*dr2);
                *Fx -= dV*dx;
                *Fy -= dV*dy;
                *Fz -= dV*dz;
            }
        }
    }
}



/*
 * Calculate the potential energy between the particles and the wall
 * 
*/

double wallsEnergy(const double *r, const double *W, double L, double Lz)  
{
    double V = 0.0;
    double dx, dy, dz, dr2, dr6;
    double dw = L/M;
    
     for (int i=0; i<M; i++) 
     {
        for (int j=0; j<M; j++) 
        {
            int m = j + i*M;
            for (int n=0; n<N; n++)  
            {
                dx = r[3*n] - i*dw - dw/2;
                dx = dx - L*rint(dx/L);
                dy = r[3*n+1] - j*dw - dw/2;
                dy = dy - L*rint(dy/L);
                dz = r[3*n+2] + Lz/2;
                dz = dz - Lz*rint(dz/Lz);
                dr2 = dx*dx + dy*dy + dz*dz;
        
                if (dr2 < L*L/4)
                {
                    dr6 = dr2*dr2*dr2;
                    V += W[2*m]/(dr6*dr6) - W[2*m+1]/dr6;
                }
            }
        }
    }
    return V*4;
}


double wallsPressure(const double *r, const double * W, double L, double Lz)
{
    double P = 0.0;
    double dx, dy, dz, dr2, dr6;
    double dw = L/M;
    
    for (int i=0; i<M; i++) 
    {
        for (int j=0; j<M; j++) 
        {
            int m = j + i*M;
            for (int n=0; n<N; n++)  
            {
                dx = r[3*n] - i*dw - dw/2;
                dx = dx - L*rint(dx/L);
                dy = r[3*n+1] - j*dw - dw/2;
                dy = dy - L*rint(dy/L);
                dz = r[3*n+2] + L/2;
                dz = dz - Lz*rint(dz/Lz);
                dr2 = dx*dx + dy*dy + dz*dz;
            
                if (dr2 < L*L/4)
                {
                    dr6 = dr2*dr2*dr2;
                    P += 24.0*W[2*m+1]/dr6 - 48.0*W[2*m]/(dr6*dr6);
                }
            }
        }
    }
    return -P/(3*L*L*Lz);
}




/*
 * Divides the volume in N voxels and stores the number of particles in each voxel.
 * returns a N/4 array containing the number of particles in each block, iterating in the z, then y, then x direction.
 * D isn't reinitialized, so it can be used for cumulative counting.
 * 
 */ // Attualmente i blocchi sono dei parallelepipedi di dimensione costante. TODO

void localDensity(const double *r, double L, double Lz, int Nv, unsigned long int *D)
{
    double * p = malloc(3*N * sizeof(double));
    memcpy(p, r, 3*N * sizeof(double));
    
    // shift the particles positions by L/2 for convenience
    for (int n=0; n<N; n++) {
        p[3*n] = p[3*n] + L/2;
        p[3*n+1] = p[3*n+1] + L/2;
        p[3*n+2] = p[3*n+2] + Lz/2;
    }
    
    int Nl = (int) rint(cbrt(Nv)); // number of cells per dimension
    if ( !isApproxEqual((double) Nl, cbrt(Nv)) )
        printf("The number passed to localDensity() should be a perfect cube, got %f != %f\n", cbrt(Nv), (double) Nl);
    
    int v;  // unique number for each triplet i,j,k
    double dL = L / Nl;
    double dLz = Lz / Nl;
    
    for (int i=0; i<Nl; i++)    {
        for (int j=0; j<Nl; j++)    {
            for (int k=0; k<Nl; k++)    {
                v = i*Nl*Nl + j*Nl + k;
                for (int n=0; n<N; n++)        {
                    if ((p[3*n]>i*dL && p[3*n]<(i+1)*dL) &&  (p[3*n+1]>j*dL && p[3*n+1]<(j+1)*dL)
                        && (p[3*n+2]>k*dLz && p[3*n+2]<(k+1)*dLz))
                        D[v]++;
                }
            }
        }
    }
    free(p);
}



/*
 * Calculate the autocorrelation function
 * 
*/

void fft_acf(const double *H, size_t length, int k_max, double * acf)   
{
    if (length < k_max*2+1) {
        perror("error: number of datapoints too low to calculate autocorrelation");
        //return;
    }
    
    fftw_plan p;
    fftw_complex *fvi, *C_H, *temp;
    int lfft = (int)(length/2)+1;
    double * Z = fftw_malloc(length * sizeof(double));
    fvi = fftw_malloc(lfft * sizeof(fftw_complex));
    temp = fftw_malloc(lfft * sizeof(fftw_complex));
    C_H = fftw_malloc(lfft * sizeof(fftw_complex));
    
    double meanH = mean(H, length);
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;

    p = fftw_plan_dft_r2c_1d(lfft, Z, fvi, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<lfft; i++)  // compute the abs2 of the transform (power spectral density of Z)
        temp[i] = fvi[i] * conj(fvi[i]) + 0.0I;
    
    p = fftw_plan_dft_1d(lfft, temp, C_H, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<k_max; i++)
        acf[i] = creal(C_H[i]) / creal(C_H[0]);


    fftw_destroy_plan(p);
    fftw_free(fvi); fftw_free(C_H); fftw_free(temp); free(Z);
}


void simple_acf(const double *H, size_t length, int k_max, double * acf)   
{
    if (length < k_max*2)
        perror("error: number of datapoints too low to calculate autocorrelation");
    
    double C_H_temp;
    double * Z = malloc(length * sizeof(double));
    double meanH = mean(H, length);
    
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;
    
    
    for (int k=0; k<k_max; k++) {
        C_H_temp = 0.0;
        
        for (int i=0; i<length-k_max-1; i++)
            C_H_temp += Z[i] * Z[i+k];

        acf[k] = C_H_temp/(length-k_max); // ci vuole il -k_max?
    }
    
    for (int k=k_max-1; k>=0; k--)
        acf[k] = acf[k] / acf[0]; // normalized autocorrelation function

    free(Z);
}






/*
    Simple math
*/

inline double sum(const double * A, size_t length)   
{
    double s = 0.;
    for (int i=0; i<length; i++)
        s += A[i];

    return s;
}

inline int intsum(const int * A, size_t length)   
{
    int s = 0;
    for (int i=0; i<length; i++)
        s += A[i];

    return s;
}


inline double dot(const double * A, double * B, size_t length)  
{
    double result = 0.0;

    for (int i=0; i<length; i++)
        result += A[i]*B[i];

    return result;
}

inline void elforel(const double * A, const double * B, double * C, size_t length)  
{
    for (int i=0; i<length; i++)
        C[i] = A[i]*B[i];
}

inline double mean(const double * A, size_t length) 
{
    return sum(A,length)/length;
}

inline double intmean(const int * A, size_t length) 
{
    int s = 0;
    for (int i=0; i<length; i++)
        s += A[i];
    
    return (double)s/length;
}

inline void zeros(size_t length, double *A)
{
    for (int i=length; i!=0; i--)
        A[i] = 0.0;
}

inline double variance(const double * A, size_t length)
{
    double * A2 = malloc(length * sizeof(double));
    elforel(A,A,A2,length);
    double var = mean(A2,length) - mean(A,length)*mean(A,length);
    free(A2);
    return var;
}

inline double variance_corr(const double * A, double tau, size_t length)
{
    double var = 0.0;
    double mean_A = mean(A,length);
    int tauint = (int) (floor(tau));
    int newlength = (int) floor(length/tauint);
    
    if (newlength < 1000)
        printf("\nThere doesn't seem to be enough data to compute the variance\n");
    
    for (int i = 0; i < newlength; i++)
        var += (A[i*tauint] - mean_A)*(A[i*tauint] - mean_A);
    
    return var/(newlength-1);
}


/*
 * Misc functions
 * 
 */

inline bool isApproxEqual(double a, double b)
{
    if (fabs(a-b) < 1e-12)
        return true;
    else
        return false;
}


int * currentTime()
{
    time_t now;
    struct tm *now_tm;
    static int currenttime[2];

    now = time(NULL);
    now_tm = localtime(&now);
    currenttime[0] = now_tm->tm_hour;
    currenttime[1] = now_tm->tm_min;
    
    return currenttime;
}


void make_directory(const char* name) 
{
    struct stat st = {0};
    
    #ifdef __linux__
       if (stat(name, &st) == -1) { mkdir(name, 0777); }
    #else
       _mkdir(name);
    #endif
}


void print_path()
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working dir: %s\n", cwd);
    } else {
        perror("getcwd() error");
    }
}
    

 
