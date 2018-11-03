using Statistics, CSVFiles, DataFrames, Printf, PyCall, ProgressMeter, FFTW
pygui(:qt); import PyPlot: pygui
pygui(:qt); import PyPlot: pygui
using Plots
pygui(:qt); :qt
pyplot()
PyPlot.pygui(true)
#plotlyjs(size=(800,600))

#run(`clang -Wall -lm -lfftw3 -O3 -march=native ./SMC_noMPI.c -o smc`)
#run(`./smc`)


function make3Dplot(A::Array{Float64}; T = -1.0, rho = -1.0, reuse=true)
    #Plots.default(size=(800,600))
    N = Int(length(A)/3)
    if rho == -1.0
        Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue,Plots.stroke(0)),
         w=7, xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false, reuse=reuse)
    else
        L = cbrt(N/rho)
        Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue, Plots.stroke(0)),
         w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false, reuse=reuse)
    end
end

function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, rho=-1.0, save=true, reuse=true)
    #Plots.default(size=(800,600))
    N = Int(size(M,1)/3)
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    X = M[1:3:3N,1]
    #pick the particles near the plane x=a/4
    I = findall(abs.(X .-a/4) .< a/4)
    # now the X indices of M in the choosen plane are 3I-2
    scatter(M[3I.-1,1], M[3I,1], m=(7,0.7,:red,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)),
     yaxis=("y",(-L/2,L/2)), leg=false)
    for i =2:size(M,2)
        scatter!(M[3I.-1,i], M[3I,i], m=(7,0.05,:blue,Plots.stroke(0)), markeralpha=0.05)
    end
    file = string("./Plots/temporal2D_",N,"_T",T,"_d",rho,".pdf")
    save && savefig(file)
end

function acf(H::Array{Float64,1}, k_max::Int64)
    C_H = zeros(k_max)
    Z = H .- mean(H)
    if k_max>20000
        bar = Progress(k_max, dt=1.0, desc="Calculating acf...", barglyphs=BarGlyphs("[=> ]"), barlen=45)
    end
    @fastmath for k = 1:k_max
        C_H_temp = 0.0
        for i = 1:length(H)-k_max-1
            @inbounds C_H_temp += Z[i] * Z[i+k-1]
        end
        C_H[k] = C_H_temp
        (k_max>20000) && next!(bar)
    end
    CH1 = C_H[1]/(length(H))
    return C_H ./ (CH1*length(H))
end

function fft_acf(H::Array{Float64,1}, k_max::Int)

    Z = H .- mean(H)
    fvi = rfft(Z)
    acf = ifft(fvi .* conj.(fvi))
    acf = real.(acf)

    return acf[1:k_max]./acf[1]
end


N = 32
M = 3
L = 16
Lz = 28
rho = round(Int, 10000*N/(L*L*Lz)) / 10000
T = 0.7

parameters = @sprintf "_N%d_M%d_r%0.4f_T%0.2f" N M rho T;
cd(string("./Data/data", parameters))
dfp = DataFrame(load(string("./positions", parameters, ".csv")))
dfw = DataFrame(load(string("./wall", parameters, ".csv")))
dfd = DataFrame(load(string("./data", parameters, ".csv")))
C_H = DataFrame(load(string("./autocorrelation", parameters, ".csv")))
lD = DataFrame(load(string("./localdensity", parameters, ".csv")))
sum(lD.n) // length(dfd.E) # check sul numero totale di particelle raccolte

lD[:n] = lD[:n] / length(dfd.E)

## local density plot


## Plot a configuration in 3D
#X0, a = MCs.initializeSystem(N, cbrt(320))
X0 = [dfp[148776, col] for col in 1:3N] # subset of columns
make3Dplot(X0, rho=rho, T=T, reuse=false)
gui()

## Check energy
Plots.plot(dfd.E[1:100:end], reuse=false, legend=false)
gui()
plot(C_H[1][1:25000], legend=false)
gui()

#acfsimple = acf(dfd.E, 10000)
#acffast = fft_acf(dfd.E, 5000)
#tausimple = sum(acfsimple)
#tau = sum(acffast)
