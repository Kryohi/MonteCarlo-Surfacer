using Statistics, CSVFiles, DataFrames, Printf, PyCall, ProgressMeter, FFTW
pygui(:qt); import PyPlot: pygui
using Plots, StatPlots
pygui(:qt); :qt
pyplot(size=(800, 600))
fnt = "helvetica"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14),
 legendfont=Plots.font(fnt,14))
PyPlot.pygui(true)
#plotlyjs(size=(800,600))

#run(`clang -Wall -lm -lfftw3 -O3 -march=native ./SMC_noMPI.c -o smc`)
#run(`./smc`)


function make3Dplot(A::Array{Float64,1}; T = -1.0, L = -1.0, Lz = -1.0, reuse=true)
    #Plots.default(size=(800,600))
    N = Int(length(A)/3)
    if L == -1.0
        Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue,Plots.stroke(0)),
         w=7, xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false, reuse=reuse)
    else
        Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue, Plots.stroke(0)),
         w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-Lz/2,Lz/2)), leg=false, reuse=reuse)
    end
end

function make3Dplot(M::DataFrame, n1, n2; T = -1.0, L = -1.0, Lz = -1.0, reuse=true)

    N = Int((size(M,2)-1)/3)
    A = [dfp[1, col] for col in 1:3N] # subset of columns
    Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue, Plots.stroke(0)),
    w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-Lz/2,Lz/2)),
    leg=false)

    for n ∈ n1+1:n2
        A = [dfp[n, col] for col in 1:3N] # subset of columns
        Plots.scatter3d!(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue, Plots.stroke(0)),
        w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-Lz/2,Lz/2)),
        leg=false)
    end
end

function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, L = -1.0, Lz = -1.0, save=true, reuse=true)
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

# makes an mp4 video made by a lot of 3D plots (can be easily modified to produce a gif instead)
# don't run this with more than ~1000 frames unless you have a lot of spare time...
function makeVideo(M; T=-1, L = -1.0, Lz = -1.0, fps = 20)
    close("all")
    Plots.default(size=(1280,1080))
    N = Int(size(M,1)/3)
    rho = round(Int, 10000*N/(L*L*Lz)) / 10000
    #rho==-1 ? L = cbrt(N/(2*maximum(M))) : L = cbrt(N/rho)
    println("\nI'm cooking pngs to make a nice video. It will take some time...")
    prog = Progress(size(M,2), dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50)  # initialize progress bar

    anim = @animate for i =1:size(M,2)
        Plots.scatter(M[1:3:3N-2,i], M[2:3:3N-1,i], M[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)), w=7,
         xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-Lz/2,Lz/2)), leg=false)
        next!(prog) # increment the progress bar
    end
    file = string("./Video/LJ",N,"_T",T,"_d",rho,".mp4")
    mp4(anim, file, fps = fps)
    gui() #show the last frame in a separate window
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

function acf_spectrum(H::Array{Float64,1}, k_max::Int)

    Z = H .- mean(H)
    fvi = rfft(Z)
    acf = real.(fvi .* conj.(fvi))

    return acf[1:k_max]
end


N = 32
M = 3
L = 30
Lz = 70
rho = round(Int, 10000*N/(L*L*Lz)) / 10000
T = 0.2

parameters = @sprintf "_N%d_M%d_r%0.4f_T%0.2f" N M rho T;
cd(string("$(ENV["HOME"])/Programming/C/MonteCarlo-Surfacer/Data/data", parameters))

dfp = DataFrame(load(string("./positions", parameters, ".csv")))
dfw = DataFrame(load(string("./wall", parameters, ".csv")))
dfd = DataFrame(load(string("./data", parameters, ".csv")))
C_H = DataFrame(load(string("./autocorrelation", parameters, ".csv")))
lD = DataFrame(load(string("./localdensity", parameters, ".csv")))
sum(lD.n) // length(dfd.E) # check sul numero totale di particelle raccolte

#lD[:n] = lD[:n] / length(dfd.E)

## local density plot
nd = Int(cbrt(length(lD.n)))
LD_impilata = zeros(nd,nd)
LD_parz_impilata = zeros(nd,nd,7)

for i = 0:nd-1
    for j = 0:nd-1
        v = i*nd*nd + j*nd;
        LD_impilata[i+1, j+1] = sum(lD.n[ (lD.nx .== i) .& (lD.ny .== j) ])

        LD_parz_impilata[i+1, j+1, 1] = sum(lD.n[ v .+ (1:2) ]);
        LD_parz_impilata[i+1, j+1, 2] = sum(lD.n[ v .+ (3:5) ]);
        LD_parz_impilata[i+1, j+1, 3] = sum(lD.n[ v .+ (5:7) ]);
        LD_parz_impilata[i+1, j+1, 4] = sum(lD.n[ v .+ (8:22) ]);
        LD_parz_impilata[i+1, j+1, 5] = sum(lD.n[ v .+ (23:25) ]);
        LD_parz_impilata[i+1, j+1, 6] = sum(lD.n[ v .+ (26:28) ]);
        LD_parz_impilata[i+1, j+1, 7] = sum(lD.n[ v .+ (29:30) ]);
        #LD_parz_impilata[i+1, j+1, 2] = sum(lD.n[ (lD.nx .== i) .& (lD.ny .== j)
        #  .& ( (lD.nz .== 2) .| (lD.nz .== 3) | (lD.nz .== 4) )]);
    end
end

z_distr = zeros(7)
for n = 1:7
    z_distr[n] = sum(LD_parz_impilata[:,:,n])
end


heatmap(LD_impilata, reuse=false)
heatmap(LD_parz_impilata[:,:,7], reuse=false)
gui()

nw = M
wall = zeros(nw,nw)
for i = 0:nw-1
    for j = 0:nw-1
        wall[i+1, j+1] = dfw.ymin[findfirst( (dfw.nx .== i) .& (dfw.ny .== j) )]
    end
end

heatmap(wall, reuse=false)

## Plot a configuration in 3D
#X0, a = MCs.initializeSystem(N, cbrt(320))
X0 = [dfp[1, col] for col in 1:3N] # subset of columns
make3Dplot(X0, L=L, Lz=Lz, T=T, reuse=false)

make3Dplot(dfp, 7000, 7010, L=L, Lz=Lz, T=T, reuse=false)


## Check energy
Plots.plot(dfd.E[1:100:end], reuse=false, legend=false)
gui()
plot(C_H[1][1:50000], legend=false)
kmax = round(Int, length(dfd.E)/2)
plot(1:kmax, acf_spectrum(dfd.E, kmax), xaxis = (:log10, (1,kmax)),
 yaxis = (:log10, (1,Inf)))
gui()

#acfsimple = acf(dfd.E, 10000)
#acffast = fft_acf(dfd.E, 5000)
#tausimple = sum(acfsimple)
#tau = sum(acffast)
