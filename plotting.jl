using Statistics, CSVFiles, DataFrames, Plots
pyplot()

run(`gcc -Wall -lm -lfftw3 -O3 -march=native ./SMC_noMPI.c -o smc`)
run(`./smc`)

dfp = DataFrame(load("positions_N32_M40_r0.10_T0.40.csv"))
dfw = DataFrame(load("wall_N32_M40_r0.10_T0.40.csv"))
N = Int((size(dfp,2)-1)/3)

X0 = [dfp[2000, col] for col in 1:3N] # subset of columns
#X0, a = MCs.initializeSystem(N, cbrt(320))
make3Dplot(X0, rho=0.1, T=0.4)
gui()

dfd = DataFrame(load("data_N32_M40_r0.10_T0.40.csv"))
plot(dfd.E[1:50:end])
gui()
acfsimple = MCs.acf(dfd.E, 60000)
acffast = MCs.fft_acf(dfd.E, 60000)
tausimple = sum(acfsimple)*20
tau = sum(acffast)*20




function make3Dplot(A::Array{Float64}; T = -1.0, rho = -1.0)
    Plots.default(size=(800,600))
    N = Int(length(A)/3)
    if rho == -1.0
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7,
         xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false)
    else
        L = cbrt(N/rho)
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7,
         xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    gui()
end

function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, rho=-1.0, save=true)
    Plots.default(size=(800,600))
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
    gui()
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
    CH1 = C_H[1]/(length(H))    # togliere o non togliere k_max, questo è il dilemma

    return C_H ./ (CH1*length(H))    # unbiased and normalized autocorrelation function
end

function fft_acf(H::Array{Float64,1}, k_max::Int)

    Z = H .- mean(H)
    fvi = rfft(Z)
    acf = ifft(fvi .* conj.(fvi))
    acf = real.(acf)

    return acf[1:k_max]./acf[1]
end
