using LinearAlgebra
using JSON
using HDF5, JLD
using Random
using TensorKit
using MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg")

include("build_mpo.jl")


#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=1;#number of virtual modes per bond
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
#init_state=nothing


W=load(init_state)["W"];
E0=load(init_state)["E0"];
Vf=ℂ[FermionNumber](0=>1, 1=>1);
U2=unitary(fuse(Vf ⊗ Vf), Vf ⊗ Vf)


A_set=create_vaccum(2);
mpo_set=build_mpo_set(2,[0.6,0.4]);

A_set_new=mpo_mps(mpo_set,A_set)

@tensor A_total[:]:=A_set_new[1][-1,1,-2]*A_set_new[2][1,-4,-3]