
/* TODO
 * 
 * Raggio LJ più corto con correzioni di coda
 * fixare LCA (memory leak?)
 * voxel cubici vicino a pareti e gap grosso in mezzo?
 * localDensity con numero minore di divisioni lungo z, più fino vicino a pareti
 * LCA diviso lungo z
 * acceptance ratio locale (per layer lungo z?)
 * decidere cosa mettere come macro e cosa come variabile passata a simulation (tipo gather_lapse)
 * legare Elegame e sigmaElegame a T
 * Unire calcolo energia e forze interparticellari e con parete?
 */

#include "SMC.h"
#ifndef rank
#define rank 0
#endif


struct Sim sMC(double L, double Lz, double T, double A, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps)   
{
    // System properties
    double rho = N / (L*L*Lz);
    
    // Data-harvesting parameters
    int gather_steps = (int)(maxsteps/gather_lapse);
    int Nc = Ncx * Ncx * Ncx;   // total number of cells for local data
    //double z_cells[Ncz];        // non-uniform range of cells along z, with box translated to (L/2,L/2,Lz/2) center
    //createZRange(Lz, z_cells);

    bool savePositions = false;
    if (gather_steps > 200000)  {
        savePositions = false; printf("Puntual positions will not be saved.\n");
    }
    
    // other variables
    clock_t start, end;
    double sim_time;
    srand(time(NULL)); // different for each process
    
    
    //copy the initial positions R0 (common to all the simulations) to the local array R
    double *R = malloc(3*N * sizeof(double));
    memcpy(R, R0, 3*N * sizeof(double));
    double * Rn = calloc(3*N, sizeof(double)); // contains the proposed particle positions
    double * E = calloc(maxsteps+1, sizeof(double));   // da sostituire a E una volta verificato funzionamento
    E[0] = energy(R, L) + wallsEnergy(R, W, L, Lz);
    double * P = calloc(gather_steps, sizeof(double));
    int * jj = calloc(maxsteps, sizeof(int));   //stores acceptance ratio
    int * jt = calloc(eqsteps, sizeof(int)); // same as above but for thermalization
    int * Rbin = calloc(N, sizeof(int));    // stores in which cell each particle is
    unsigned long * lD = calloc(Nc, sizeof(unsigned long)); // local density
    unsigned long * lD_old = calloc(Nc, sizeof(unsigned long));
    unsigned long * Mu = calloc(Nc, sizeof(unsigned long)); // local mobility
    unsigned long * Mu_old = calloc(Nc, sizeof(unsigned long));
    int * clusters_global = calloc((int)(3*(N*N-N)/2), sizeof(int));
    double l1 = 0.0;
    double l2[7];  // stores number of occurrencies of a certain second cluster number
    double l3[7];  // stores number of occurrencies of a certain third cluster number
    
    
    // Initialize csv files
    char filename[64];
    
    snprintf(filename, 64, "./positions_N%d_M%d_r%0.4f_T%0.2f_rank%d.csv", N, M, rho, T, rank);
    FILE *positions = fopen(filename, "w");
    for (int n=0; n<N; n++)
        fprintf(positions, "x%d,y%d,z%d,", n+1, n+1, n+1);
    fprintf(positions, "\n");
    for (int i=0; i<3*N; i++)
        fprintf(positions, "%0.3lf,", R[i]);    // provare %6g
    fprintf(positions, "\n");
    
    snprintf(filename, 64, "./data_N%d_M%d_r%0.4f_T%0.2f_rank%d.csv", N, M, rho, T, rank);
    FILE *data = fopen(filename, "w");
    fprintf(data, "E, P, jj\n");
    
    FILE * local;
    snprintf(filename, 64, "./local_N%d_M%d_r%0.4f_T%0.2f_rank%d.csv", N, M, rho, T, rank);
    local = fopen(filename, "w");
    fprintf(local, "nx, ny, nz, n, mu\n");
    
    FILE * local_temp;
    snprintf(filename, 64, "./local_temp_N%d_M%d_r%0.4f_T%0.2f_rank%d.csv", N, M, rho, T, rank);
    local_temp = fopen(filename, "w");    
    fprintf(local_temp, "nx, ny, nz, n, mu\n");
    
    FILE * total_clusters;
    snprintf(filename, 64, "./total_clusters_N%d_M%d_r%0.4f_T%0.2f_rank%d.csv", N, M, rho, T, rank);
    total_clusters = fopen(filename, "w");    
    fprintf(total_clusters, "l1, l2, l3\n");
    
    FILE * autocorrelation;
    snprintf(filename, 64, "./autocorrelation_N%d_M%d_r%0.4f_T%0.2f_rank%d.csv", N, M, rho, T, rank);
    autocorrelation = fopen(filename, "w");
    fprintf(autocorrelation, "CH\n");
    
    if (autocorrelation == NULL || positions == NULL || data == NULL || local == NULL)
        perror("error while opening csv files");

    
    printf("\nStarting new run with %d particles in %0.1fx%0.1fx%0.1f box, ", N, L, L, Lz);
    printf("T=%0.2f, rho=%0.4f, A=%0.3f, for %d steps...\n", T, rho, A, maxsteps);

    
    
    /*  Thermalization   */
    
    A = A*2;    // to help the fluid reaching the equilibrium a bigger A is used, because the particles start far from the walls.
    start = clock();
    
    for (int n=0; n<eqsteps; n++)
    {
        //E[n] = energy(R, L) + wallsEnergy(R, W, L, Lz);// calcolata in modo più intelligente dentro oneParticleMoves
        E[n+1] = E[n];  // then the energy difference gets added inside the function
        oneParticleMoves(R, Rn, W, L, Lz, A, T, &jt[n], &E[n+1]);
    }
    
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    int *now = currentTime();
    printf("\nThermalization completed in %0.1f mins at %02d:%02d, with ", sim_time/60, now[0], now[1]);
    printf("average acceptance ratio %0.3f, mean energy %0.3f.\n", intmean(jt,eqsteps)/N, mean(E,eqsteps)+3*N*T/2);
    A = A/2;
        
    
    
    /*  Actual simulation   */
    
    printf("The expected time of execution is ~%0.1f minutes.\n", 1.03*sim_time*maxsteps/eqsteps/60);
    start = clock();

    for (int n=0; n<maxsteps; n++)
    {
        // All data except energy is computed and stored at gather_lapse (or more) intervals
        if ((n+1) % gather_lapse == 0)  {
            int k = (int)((n+1)/gather_lapse);
            
            //P[k] = pressure(R, L, Lz) + wallsPressure(R, W, L, Lz);
            localDensityAndMobility(R, L, Lz, lD, Rbin, Mu);
            
            if (k % LCA_TIME == 0)
            {   // sarebbe da calcolare in modo simile a local density, solo con strati lungo z
                clusterAnalysis(R, N, L, clusters_global);
                for (int i=0; i<(N*N-N)/2; i++)  
                { 
                    if (clusters_global[3*i] !=0)   {
                        l1 += 1 / (gather_steps/LCA_TIME);
                        if (clusters_global[3*i+1] !=0) printf("\nlca 2 = %d !", clusters_global[3*i+1]);
                        if (clusters_global[3*i+2] !=0) printf("\nlca 3 = %d !", clusters_global[3*i+2]);
                        l2[clusters_global[3*i+1]] += 1 / (gather_steps/LCA_TIME);
                        l3[clusters_global[3*i+2]] += 1 / (gather_steps/LCA_TIME);
                    }
                }
            }
            
            if (k % STORAGE_TIME == 0)  
            {   
                if (!savePositions) // always save a few configurations
                {
                    for (int i=0; i<3*N; i++)
                        fprintf(positions, "%0.3lf,", R[i]);    // provare %6g
                    fprintf(positions, "\n");
                }
                
                // dump of the local density in the last million or so steps
                printf("\rStoring the latest density distribution at %d steps... ", n+1);
                fflush(stdout);
                
                for (int i=0; i<Ncx; i++)    {
                    for (int j=0; j<Ncx; j++)    {
                        for (int k=0; k<Ncz; k++)    {
                            int v = i*Ncx*Ncz + j*Ncz + k;
                            fprintf(local_temp, "%d, %d, %d, %lu, %lu\n", i, j, k, lD[v] - lD_old[v], Mu[v] - Mu_old[v]);
                        }
                    }
                }
                // save the current local density and mobility cumulative numbers
                memcpy(lD_old, lD, Nc * sizeof(unsigned long)); 
                memcpy(Mu_old, Mu, Nc * sizeof(unsigned long));
            }
            
            if (savePositions)  {
                for (int i=0; i<3*N; i++)
                    fprintf(positions, "%0.3lf,", R[i]);    // provare %6g

                fprintf(positions, "\n");
            }
            // check that all particles are within the walls
            boundsCheck(R, L, Lz-0.1);
        }
        
        E[n+1] = E[n];  // then the energy difference gets added inside the function
        oneParticleMoves(R, Rn, W, L, Lz, A, T, &jj[n], &E[n+1]);
    }
    
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\n\nTime: %0.1f s (%0.1f per million)\n", sim_time, sim_time*1e6/maxsteps);

    
    
    /*  Data preparation and storage  */

    // total energy and total pressure
    for (int k=0; k<gather_steps; k++)
        P[k] += rho*T;
    
    for (int n=0; n<maxsteps+1; n++)
        E[n] += 3*N*T/2;
    
    // save temporal data of the system (gather_steps arrays of energy, pressure and acceptance ratio)
    for (int k=0; k<gather_steps; k++)
        fprintf(data, "%0.9lf, %0.9lf, %d\n", E[k*gather_lapse], P[k], jj[k]);
    
    
    for (int i=0; i<Ncx; i++)    {
        for (int j=0; j<Ncx; j++)    {
            for (int k=0; k<Ncz; k++)    {
                int v = i*Ncx*Ncz + j*Ncz + k;
                fprintf(local, "%d, %d, %d, %lu, %lu\n", i, j, k, lD[v], Mu[v]);
            }
        }
    }
    // temporaneo
    printf("l1[1] = %0.9f\n", l1);
    printf("l2[0] = %0.9f\tl2[1] = %0.9f\tl2[2] = %0.9f\tl2[3] = %0.9f\tl2[4] = %0.9f\tl2[5] = %0.9f\n",
           l2[0], l2[1], l2[2], l2[3], l2[4], l2[5]);
    printf("l3[0] = %0.9f\tl3[1] = %0.9f\tl3[2] = %0.9f\tl3[3] = %0.9f\tl3[4] = %0.9f\tl3[5] = %0.9f\n",
           l3[0], l3[1], l3[2], l3[3], l3[4], l3[5]);

    // autocorrelation calculation
    DoubleArray acf = fft_acf(E, maxsteps+1, KMAX);
    double tau = sum(acf.data, acf.length);
    //simple_acf(E, maxsteps+1, kmax, acf2);    // da eliminare dopo aver confrontato
    //printf("TauSimple: %f \n", tau);
    
    for (int m=0; m<acf.length; m++)
      fprintf(autocorrelation, "%0.6lf\n", acf.data[m]);
    
    // Create struct of the mean values and deviations to return
    Sim results;
    results.E = mean(E, maxsteps+1);
    results.dE = sqrt(variance(E, maxsteps+1));
    results.P = mean(P, gather_steps);
    results.dP = sqrt(variance(P, gather_steps));
    results.acceptance_ratio = intmean(jj, maxsteps)/N;
    results.tau = tau;  // * gather_lapse;
    results.cv = variance(E, maxsteps+1) / (T*T);
    memcpy(results.Rfinal, R, 3*N * sizeof(double));
    for (int s=0; s<7; s++) {
        results.l2[s] = l2[s];
        results.l3[s] = l3[s];
    }
    results.ACF = acf;
   
    // free the allocated memory
    free(R); free(Rn); free(E); free(P); free(jj); free(jt); free(Rbin);// free(acf);
    free(lD); free(Mu); free(lD_old); free(Mu_old); free(clusters_global);
    printf("gnam6\n");
    fclose(positions); fclose(data); fclose(autocorrelation); fclose(local); fclose(local_temp);
    fclose(total_clusters);
    printf("gnam7\n");

    return results;
}



/*
 * Executes a single particle Smart Monte Carlo step for each of the N particles.
 * Each times it starts the loop from a random particle (not sure if this is useful...)
 * It modifies the passed arrays R, Rn and j (the last one containing the ratio of accepted steps).
 * 
*/

void oneParticleMoves(double * R, double * Rn, const double * W, double L, double Lz, double A0, double T, int *j, double *E)
{
    double * displ = malloc(3*N * sizeof(double));
    double Um, Un, deltaX, deltaY, deltaZ, Fmx, Fmy, Fmz, Fnx, Fny, Fnz, deltaW, ap;
    double A = A0;
        
    vecBoxMuller(sqrt(2.0*A), 3*N, displ);
    
    for (int i=0; i<3*N; i++)   // controllare se è necessario qua o si può usare solo una volta all'inizio
        Rn[i] = R[i];
    
    // at each oneParticleMoves call, we start moving a different particle (offset % N) // not used right now
    int n; int offset = rand();
    
    for (int nn=0; nn<N; nn++)
    {
        n = (nn+offset)%N;
        //n = nn;
        
        //(fabs(R[3*n+2]) > Lz/8) ? (A=A0) : (A=A0*2); // if the particle is in the middle of the box the step can be bigger

        // calculate the potential energy of particle n, first due to other particles and then to the wall
        Um = energySingle(R, L, n) + wallsEnergySingle(R[3*n], R[3*n+1], R[3*n+2], W, L, Lz);
        
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
        
        Rn[3*n] = Rn[3*n] - L*rint(Rn[3*n]/L);         // verificare che vada bene qui
        Rn[3*n+1] = Rn[3*n+1] - L*rint(Rn[3*n+1]/L);

        // calculate energy and forces in the proposed new position
        Un = energySingle(Rn, L, n) + wallsEnergySingle(Rn[3*n], Rn[3*n+1], Rn[3*n+2], W, L, Lz);
        forceSingle(Rn, L, n, &Fnx, &Fny, &Fnz);
        wallsForce(Rn[3*n], Rn[3*n+1], Rn[3*n+2], W, L, Lz, &Fnx, &Fny, &Fnz);


        // Calculate the acceptance probability for the single-particle move
        
        deltaW = ((Fnx-Fmx)*(Fnx-Fmx) + (Fny-Fmy)*(Fny-Fmy) + (Fnz-Fmz)*(Fnz-Fmz) +
            2.0*((Fnx-Fmx)*Fmx + (Fny-Fmy)*Fmy + (Fnz-Fmz)*Fmz)) * A/(4.0*T);

        ap = exp(-(Un-Um + (deltaX*(Fnx+Fmx) + deltaY*(Fny+Fmy) + deltaZ*(Fnz+Fmz))/2.0 + deltaW)/T);

        
        // Accepts the move by comparing ap with a random uniformly distributed probability
        // If the move is rejected, return Rn to the initial state (equal to R)
        
        if ((double)rand()/(double)RAND_MAX < ap)
        {
            R[3*n] = Rn[3*n];
            R[3*n+1] = Rn[3*n+1];
            R[3*n+2] = Rn[3*n+2];
            *j += 1;
            *E += Un-Um;
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
    
    double * displ = malloc(3*N * sizeof(double));
    double Um, Un, deltaX, deltaY, deltaZ, Fmx, Fmy, Fmz, Fnx, Fny, Fnz, deltaW, ap;
        
    vecBoxMuller(sqrt(2*A), 3*N, displ);
    
    for (int i=0; i<3*N; i++)   // controllare se è necessario qua o si può usare solo una volta all'inizio
        Rn[i] = R[i];

    
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
 */
void initializeBox(double L, double Lz, int N_, double *X) 
{
    srand(42);
    int Nc = (int) ceil(N_/4);  // total number of fcc cells
    int Na = 1; // number of cells along x and y
    // find largest cube of cells that doesn't hold Nc cells
    for (int nc = 1; nc < N_; nc++)
    {
        if (nc*nc*nc > Nc)   {
            Na = nc-1;
            break;
        }
    }
    int Nz = rint((N_/4)/(Na*Na));
    if ( !isPicoEqual( (N_/4)/(Na*Na), (double) Nz) )  // da testare per vedere se si può togliere
        perror("Can't make a crystal with this N, it should be an integer times a perfect square, all divisible by 4.\n");

    double a = L / Na;

    for (int i=0; i<Na; i++)    {   // loop over every cell of the fcc lattice
        for (int j=0; j<Na; j++)    {
            for (int k=0; k<Nz; k++)    {
                int n = i*Na*Nz + j*Nz + k; // unique number for each triplet i,j,k
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

    for (int n=0; n<N; n++) {    // avoid particles exactly at the edges of the box or overlapping
        X[3*n] += a/4 + L * (rand()/RAND_MAX)/50;
        X[3*n+1] += a/4 + L * (rand()/RAND_MAX)/50;
        X[3*n+2] += a/4 + L * (rand()/RAND_MAX)/50;
    }

    shiftSystem3D(X,L,Lz-Lz/20.0);   // uses L instead of Lz in order to put the lattice at the center of the box
    if ( boundsCheck(X, L, Lz-0.5) > 0 )
        perror("Lz is too small or there is something else going wrong\n");

}


/*
 Takes average and standard deviation of the distance from the y-axis (at f(x)=0) and of the maximum binding energy
 and puts in the W array two gaussian distributions of resulting parameters "a" and "b".
 These parameters enter the Lennard-Jones potential in the form V = 4*(a/r^12 - b/r^6).
*/
// Per ora divide la parete in M fettine rettangolari, e il potenziale verrà generato da una linea per ogni fettina 

void initializeWalls(double x0m, double x0sigma, double ymm, double ymsigma, double *W, FILE * wall)    
{
    srand(42);  // necessary because initializeWalls gets called in main, otherwise the walls are always the same
    
    int m;
    double x0;
    double * X0 = malloc(M*M * sizeof(double));
    double * YM = malloc(M*M * sizeof(double));
    
    vecBoxMuller(x0sigma, M*M, X0);
    vecBoxMuller(ymsigma, M*M, YM);
    
    // saves the parameter distribution to a file (the rest of the program only uses a and b instead)
    fprintf(wall, "nx, ny, x0, ymin\n");
    
    for (int i=0; i<M; i++) {
        for (int j=0; j<M; j++) {
            m = i*M + j;
            x0 = X0[m]+x0m;
            fprintf(wall, "%d, %d, %f, %f\n", i, j, x0, YM[m]+ymm);
            W[2*m] = pow(x0, 12.0) * (YM[m]+ymm);     // a
            W[2*m+1] = pow(x0, 6.) * (YM[m]+ymm);     // b
        }
    }
    
    free(X0); free(YM); fclose(wall);
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

inline void shiftSystem2D(double *r, double L)
{
    for (int j=0; j<N; j++) {
        r[3*j] = r[3*j] - L*rint(r[3*j]/L);
        r[3*j+1] = r[3*j+1] - L*rint(r[3*j+1]/L);
    }
}

inline int boundsCheck(double *r, double L, double Lz)
{
    int out = 0;
    for (int j=0; j<N; j++) {
        if ((fabs(r[3*j]) > L/2.0) || (fabs(r[3*j+1]) > L/2.0))
        {
            printf("Particles are escaping the system and going to the beta-carotene Valhalla\n");
            out++;
        }
        else if (fabs(r[3*j+2]) > Lz/2.0) {
            printf("Particles are smashing the walls :(\n");
        }
    }
    return out;
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
            
            if (dr2 < L*L/4.0)
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
            dx = r[3*i] - r[3*l];
            dx = dx - L*rint(dx/L);
            dy = r[3*i+1] - r[3*l+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*i+2] - r[3*l+2];
            //dz = dz - Lz*rint(dz/Lz); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4.0)
            {
                dr8 = dr2*dr2*dr2*dr2;
                dV = 48.0/(dr8*dr2*dr2*dr2) - 24.0/dr8 ;    // -(dV/dr) / dr
                *Fx += dV*dx;   // se si devono attrarre (-dV) è negativo 
                *Fy += dV*dy;
                *Fz += dV*dz;
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
    double dx, dy, dz, dr2, dr6, dz6;
    double dw = L/M;    // distance between two wall elements
    
    dz = rz + Lz/2;
    dz = dz - Lz*rint(dz/Lz);
    if (rz <= -Lz/2.0) dz = 0.0001;
    else if (rz >= Lz/2) dz = -0.0001;
    dz6 = dz*dz*dz*dz*dz*dz;
    V += a0/(dz6*dz6) - b0/dz6;
    
    for (int i=0; i<M; i++) 
    {
        for (int j=0; j<M; j++) 
        {
            int m = j + i*M;
            dx = rx - i*dw;// - dw/2;
            dx = dx - L*rint(dx/L);
            dy = ry - j*dw;// - dw/2;
            dy = dy - L*rint(dy/L);
            
            dr2 = dx*dx + dy*dy + dz*dz;
        
            if (dr2 < L*L/4.0)
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

void wallsForce(double rx, double ry, double rz, const double * W, double L, double Lz, double *Fx, double *Fy, double *Fz) 
{ 
    double dx, dy, dz, dr2, dr8, dz8, dV;
    double dw = L/M;    // distance between consecutive wall potential sources
    
    // se rz è positivo, rint dà 1 e la distanza è calcolata da parete sopra. 
    // Infatti dz = (rz-L/2) < 0, forza "in direzione" delle z negative
    // se rz è negativo, dz = (rz+L/2) > 0
    // TODO controllare segno e/o casi in cui potrebbe dare risultati non voluti
    dz = rz + Lz/2.0;
    dz = dz - Lz*rint(dz/Lz);
    if (rz <= -Lz/2.0) dz = 0.0001;
    else if (rz >= Lz/2.0) dz = -0.0001;
    dz8 = dz*dz*dz*dz*dz*dz*dz*dz;
    dV = 48.0 * a0 / (dz8*dz*dz*dz*dz*dz*dz) - 24.0 * b0 / dz8;
    *Fz += dV*dz;
    
    for (int i=0; i<M; i++) 
    {
        for (int j=0; j<M; j++) 
        {
            int m = j + i*M;
            dx = rx - i*dw;// - dw/2;
            dx = dx - L*rint(dx/L);
            dy = ry - j*dw;// - dw/2;
            dy = dy - L*rint(dy/L);
            
            dr2 = dx*dx + dy*dy + dz*dz;
        
            if (dr2 < L*L/4)
            {
                dr8 = dr2*dr2*dr2*dr2;
                dV = 48.0 * W[2*m] / (dr8*dr2*dr2*dr2) - 24.0 * W[2*m+1] / dr8; //(-dV/dr)/r
                *Fx += dV*dx;
                *Fy += dV*dy;
                *Fz += dV*dz;
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
    double dx, dy, dz, dr2, dr6, dz6;
    double dw = L/M;
    
    for (int n=0; n<N; n++)  
    {
        dz = r[3*n+2] + Lz/2;
        dz = dz - Lz*rint(dz/Lz);
        if (r[3*n+2] <= -Lz/2) dz = 0.0001;
        else if (r[3*n+2] >= Lz/2) dz = -0.0001;
        dz6 = dz*dz*dz*dz*dz*dz;
        V += a0/(dz6*dz6) - b0/dz6;
        
        for (int i=0; i<M; i++) 
        {
            for (int j=0; j<M; j++) 
            {
                int m = j + i*M;
                dx = r[3*n] - i*dw;// - dw/2;
                dx = dx - L*rint(dx/L);
                dy = r[3*n+1] - j*dw;// - dw/2;
                dy = dy - L*rint(dy/L);
                
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
    double dx, dy, dz, dr2, dr6, dz6;
    double dw = L/M;
    
    for (int i=0; i<M; i++) 
    {
        for (int j=0; j<M; j++) 
        {
            int m = j + i*M;
            for (int n=0; n<N; n++)  
            {
                dx = r[3*n] - i*dw;// - dw/2;
                dx = dx - L*rint(dx/L);
                dy = r[3*n+1] - j*dw;// - dw/2;
                dy = dy - L*rint(dy/L);
                dz = r[3*n+2] + L/2;
                dz = dz - Lz*rint(dz/Lz);
                dr2 = dx*dx + dy*dy + dz*dz;
            
                if (dr2 < L*L/4)
                {
                    dr6 = dr2*dr2*dr2;
                    P += 24.0*W[2*m+1]/dr6 - 48.0*W[2*m]/(dr6*dr6);
                    dz6 = dz*dz*dz*dz*dz*dz;
                    P += 24.0*b0/dz6 - 48.0*a0/(dz6*dz6);
                }
            }
        }
    }
    return -P/(3*L*L*Lz);
}




/*
 * Divides the volume in N=Nc*Nc*Ncz voxels and stores the number of particles in each voxel.
 * returns an array containing the number of particles in each block, iterating in the z, then y, then x direction.
 * D isn't reinitialized, so it can be used for cumulative counting.
 * 
 * // Attualmente i blocchi sono dei parallelepipedi di dimensione costante.
 * Rbin vettore con N elementi, ognuno con numero v di appartenenza cella
 * Mu vettore Nc elementi, in ognuno ogni volta che un particella esce dal cubetto associato si alza un contatore
 * 
 * Much faster than the nonuniform version below
*/
void localDensityAndMobility(const double *r, double L, double Lz, unsigned long int *D, int *Rbin, unsigned long int *Mu)
{
    uint8_t i, j, k;
    int v;
    for (int n=0; n<N; n++) {
        i = floor((r[3*n]/L+.5) * Ncx);
        j = floor((r[3*n+1]/L+.5) * Ncx);
        k = floor((r[3*n+2]/Lz+.5) * Ncz);
        v = i*Ncx*Ncz + j*Ncz + k;
        D[v]++;
        if (Rbin[n] != v) {
            Mu[v]++;      // if particle n changed cell, mobility for that cell up by one
            Rbin[n] = v;  // particle n is now in cell v
        }
    }
}

// TODO: ciclo su n esterno, break da k quando trova il blocco giusto. i e j calcolati come sopra
void localDensityAndMobility_nonuniz(const double *r, double L, double Lz, double *z_cells, unsigned long int *D, int *Rbin, unsigned long int *Mu)
{
    double * p = malloc(3*N * sizeof(double));
    memcpy(p, r, 3*N * sizeof(double));
    
    // shift the particles positions by L/2 for convenience
    for (int n=0; n<N; n++) {
        p[3*n] = p[3*n] + L/2.0;
        p[3*n+1] = p[3*n+1] + L/2.0;
        p[3*n+2] = p[3*n+2] + Lz/2.0;
    }
    
    int v;  // unique number for each triplet i,j,k
    double dL = L / Ncx;
    
    for (int i=0; i<Ncx; i++)    {
        for (int j=0; j<Ncx; j++)    {
            for (int k=0; k<Ncz; k++)    {
                v = i*Ncx*Ncz + j*Ncz + k;
                for (int n=0; n<N; n++)        {
                    if ( ((p[3*n]>i*dL && p[3*n]<(i+1)*dL) && (p[3*n+1]>j*dL && p[3*n+1]<(j+1)*dL) 
                        && p[3*n+2]>z_cells[k] && p[3*n+2]<z_cells[k+1]) )   // provare a precalcolare array con i k*dL etc.
                    {
                        D[v]++; // local density counter up by one
                        if (Rbin[n] != v) {
                            Mu[v]++;        // if particle n changed cell, mobility for that cell up by one
                            Rbin[n] = v;  // particle n is now in cell v
                        }
                    }
                }
            }
        }
    }
    free(p);
}



// TODO 
// verificare che ordine indici sia il più veloce, definire LCA e fare in modo che di def in 0 sia 2
// Da aggiungere versione locale per bin vicini a superficie
void clusterAnalysis(const double *r, int N_, double L, int *LCA)
{
    double * dist2 = malloc((int)((N_*N_ -N_)/2) * sizeof(double));   // distanze sono matr. triangolare
    bool * num1 = calloc((int)((N_*N_ -N_)/2), sizeof(bool));
    int * num2 = calloc((int)((N_*N_ -N_)/2), sizeof(int));
    int * num3 = calloc((int)((N_*N_ -N_)/2), sizeof(int)); 
    int common_nn[8];
    int idx, idx2, idx3;    // indexes over each couples of particles
    double dx, dy, dz;
    
    // finds all the couples that are near each other
    for (int l=1; l<N_; l++)    
    {
        for (int i=0; i<l; i++)   
        {
            idx = (l*l-3*l+2)/2 + i;  // index is [(l-1)^2 - (l-1)]/2 + i
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dist2[idx] = dx*dx + dy*dy + dz*dz;
            if (dist2[idx] < LCA_cutoff * LCA_cutoff)
                // se sotto il cutoff, sono nn (primo numero)
                num1[idx] = 1;
                            
        }
    }
    
    // calculate the string type for each couple
    for (int l=1; l<N_; l++)    
    {
        for (int i=0; i<l; i++)   
        {
            idx = (l*l-3*l+2)/2 + i;
            if (num1[idx])
            {
                for (int i2=0; i2<l; i2++)  
                {   // search near neighbors common to both i and l
                    if (i2 != i)    // excludes the i-i couple
                    { 
                        idx2 = idx - i + i2;        // l-i2 couple
                        idx3 = (i2*i2-3*i2+2)/2 + i;  // i-i2 couple
                        if (num1[idx2] & num1[idx3]) 
                        {
                            common_nn[num2[idx]] = i2;  // saves which particles (at l,i2) are neighbors of both i and l
                            num2[idx]++;
                        }
                    }
                }
                // search if the common neighbors found are in turn near each other
                if (num2[idx] > 1)
                {
                    for  (int m = 1; m < num2[idx]; m++)    
                    {
                        idx2 = (common_nn[m]*common_nn[m] - 3*common_nn[m] +2)/2 + common_nn[m-1];   // index of the common_nn couples
                        if (num1[idx2]) 
                            num3[idx]++;
                    }
                }
            }
        }
    }
            
     
    for (int n = 0; n < (N_*N_-N_)/2; n++)
    {
        if (num2[n]>6) printf("LCA cutoff might be too big, clustering data will be corrupted\n");
        LCA[3*n+0] = (int) num1[n];
        LCA[3*n+1] = num2[n];
        LCA[3*n+2] = num3[n];
    }
    
    free(num1); free(num2); free(num3);
}




/*
 * Calculate the autocorrelation function
 * 
*/

DoubleArray fft_acf(const double *H, size_t length, int k_max)   
{
    DoubleArray acf;
    if (length < k_max*2+1) {
        k_max = (int)rint(length/2) - 2;
        printf("Number of datapoints too low to calculate autocorrelation, new k_max: %d\n", k_max);
    }
    acf.length = k_max;
    acf.data = calloc(k_max, sizeof(double));
    
    fftw_plan p;
    fftw_complex *fvi, *C_H, *temp;
    int lfft = (int)(length/2) + length%2;
    double * Z = fftw_malloc(length * sizeof(double));
    fvi = fftw_malloc(lfft * sizeof(fftw_complex));
    temp = fftw_malloc(lfft * sizeof(fftw_complex));
    C_H = fftw_malloc(lfft * sizeof(fftw_complex));
    
    double meanH = mean(H, length);
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;

    p = fftw_plan_dft_r2c_1d(length, Z, fvi, FFTW_ESTIMATE);    // length o lfft?
    fftw_execute(p);

    for (int i=0; i<lfft; i++)  // compute the abs2 of the transform (power spectral density of Z)
        temp[i] = fvi[i] * conj(fvi[i]) + 0.0I;
    
    p = fftw_plan_dft_1d(lfft, temp, C_H, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<k_max; i++)
        acf.data[i] = creal(C_H[i]) / creal(C_H[0]);


    fftw_destroy_plan(p);
    fftw_free(fvi); fftw_free(C_H); fftw_free(temp); free(Z);
    return acf;
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



/*    Additional math not in matematicose    */

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

// Create range of Ncz numbers between 0 and Lz. All the layers except the central 3 have a LAYER_DEPTH thickness
void createZRange(double Lz, double * z_cells)
{
    for (int k=0; k<(int)((Ncz-2)/2); k++)
        z_cells[k] = LAYER_DEPTH * k;
    
    for (int k=0; k<(int)((Ncz-2)/2); k++)
        z_cells[Ncz-k-1] = Lz - LAYER_DEPTH * k;
    
    double freespace = Lz - (Ncz-4)*LAYER_DEPTH;
    z_cells[(int)((Ncz-2)/2)] = Lz/2 - freespace/6;
    //z_cells[(int)((Ncz-1)/2)] = Lz/2;
    z_cells[(int)(Ncz/2)] = Lz/2 + freespace/6;
    
    //for (int k=0; k<Ncz; k++)
    //    printf("\ncellz %f ", z_cells[k]);
    //printf("\n");
}


 
