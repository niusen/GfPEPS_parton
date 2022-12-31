using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
using Combinatorics

cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg\\mpo_swap_construction")

include("mpo_mps_funs.jl");

#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond

#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
#init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
init_state="QWZ_M"*string(M)*".jld";#initialize: nothing

W=load(init_state)["W"];
E0=load(init_state)["E0"];


W_set=create_mpo(W);


mps_set=create_vaccum_mps(P+4*M);
for cc=1:Q
    mps_set=mpo_mps(W_set[cc,:],mps_set);
end

psi_G=mps_set;

overlap_1D(psi_G,psi_G)

#check the state
x1=3;
x2=5;
W1=zeros(P+4*M,1);W1[x1,1]=1;
W2=zeros(P+4*M,1);W2[x2,1]=1;
W1_set=create_mpo(W1);
W2_set=create_mpo(W2);
#conjugate
for cc=1:Int((P+4*M)/2)
    W2_set[cc]=permute(W2_set[cc],(1,2,3,4,))
    W2_set[cc]=permute(W2_set[cc]',(1,4,3,2,))
end

#C_{x1,up}^dag * C_{x1,down}^dag C_{x1,down} *C_{x1,up} 
println(overlap_1D(psi_G, mpo_mps(W1_set, mpo_mps(W2_set,psi_G))))


Correl=W*W';
println(Correl[x2,x1])






bond_coe=zeros(4,2);bond_coe[1,1]=1;bond_coe[3,1]=1;bond_coe[2,2]=1;bond_coe[4,2]=1;
bond_set=create_mpo(bond_coe);
#conjugate
for dd=1:2
for cc=1:2
    bond_set[cc,dd]=permute(bond_set[cc,dd],(1,2,3,4,))
    bond_set[cc,dd]=bond_set[cc,dd]'
end
end
vaccums=create_vaccum_mps(4);
for cc=1:2
    vaccums[cc]=permute(vaccums[cc],(1,2,3,))
    vaccums[cc]=vaccums[cc]';
end
for cc=1:2
    vaccums=mpo_mps(bond_set[cc,:],vaccums);
end


@tensor bond[:]:=vaccums[1][-1,1,-2]*vaccums[2][1,-4,-3];
U=unitary(fuse(space(bond,1)⊗space(bond,4)),space(bond,1)⊗space(bond,4));
@tensor bond[:]:=bond[1,-2,-3,2]*U[-1,1,2];



bond_m=zeros(1,1,2,2,2,2)*im;
bond_m[1,1,2,2,1,1]=1;
bond_m[1,1,1,2,2,1]=-1;
bond_m[1,1,2,1,1,2]=1;
bond_m[1,1,1,1,2,2]=-1;
bond_gate=TensorMap(bond_m,Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](0=>1, 1=>1) ⊗ Rep[U₁](0=>1, 1=>1), Rep[U₁](0=>1, 1=>1)' ⊗ Rep[U₁](0=>1, 1=>1)' );
bond_gate=permute(bond_gate,(1,2,3,4,5,6,))
U=unitary(fuse(space(bond_gate,1)⊗space(bond_gate,2)),space(bond_gate,1)⊗space(bond_gate,2));
@tensor bond_gate[:]:=bond_gate[1,2,-2,-3,-4,-5]*U[-1,1,2];
U=unitary(fuse(space(bond_gate,2)⊗space(bond_gate,3)),space(bond_gate,2)⊗space(bond_gate,3));
@tensor bond_gate[:]:=bond_gate[-1,1,2,3,4]*U[-2,1,2]*U[-3,3,4];

save("swap_gate_Tensor_M2.jld2", "psi_G",psi_G, "bond",bond,"bond_gate",bond_gate);