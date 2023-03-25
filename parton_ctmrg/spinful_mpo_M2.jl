using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
using Combinatorics

cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg")

include("mpo_mps_funs.jl");

#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond

#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
#init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
#init_state="QWZ_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_correct_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_correct_decoupled_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_incorrect_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_correct_modified_M"*string(M)*".jld";#initialize: nothing
init_state="Rotate_decoupled_C2_theta_0.25_M"*string(M)*".jld";#initialize: nothing

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
x1=4;
x2=5;
W1=zeros(P+4*M,1);W1[x1,1]=1;
W2=zeros(P+4*M,1);W2[x2,1]=1;
W1_set=create_mpo(W1);
W2_set=create_mpo(W2);
#conjugate
for cc=1:P+4*M
    W2_set[cc]=permute(W2_set[cc],(1,2,3,4,))
    W2_set[cc]=permute(W2_set[cc]',(1,4,3,2,))
end

#C_{x1,up}^dag * C_{x1,down}^dag C_{x1,down} *C_{x1,up} 
println(overlap_1D(psi_G, mpo_mps(W1_set, mpo_mps(W2_set,psi_G))))


Correl=W*W';
println(Correl[x2,x1]^2)






bond_coe=zeros(2,1);bond_coe[1,1]=1;bond_coe[2,1]=1;
bond_set=create_mpo(bond_coe);
#conjugate
for cc=1:2
    bond_set[cc]=permute(bond_set[cc],(1,2,3,4,))
    bond_set[cc]=bond_set[cc]';
    #bond_set[cc]=permute(bond_set[cc]',(1,4,3,2,))
end
vaccums=create_vaccum_mps(2);
for cc=1:2
    vaccums[cc]=permute(vaccums[cc],(1,2,3,))
    vaccums[cc]=vaccums[cc]';
end
for cc=1:1
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
bond_m_U1=TensorMap(bond_m,Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](0=>1, 1=>1) ⊗ Rep[U₁](0=>1, 1=>1), Rep[U₁](0=>1, 1=>1)' ⊗ Rep[U₁](0=>1, 1=>1)' );
bond_m_U1=permute(bond_m_U1,(1,2,3,4,5,6,))

bond_m=reshape(bond_m,1,4,4)
Pm=zeros(4,4)*im;Pm[1,1]=1;Pm[2,4]=1;Pm[3,2]=1;Pm[4,3]=1;
@tensor bond_m[:]:=bond_m[-1,1,2]*Pm[-2,1]*Pm[-3,2];
V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);#element order after converting to dense: <0,0>, <up,down>, <up,0>, <0,down>, 
Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
bond_gate=TensorMap(bond_m,Vdummy ⊗ V ← V');
bond_gate=permute(bond_gate,(1,2,3,));

#spin-spin correlation operator 
SS=zeros(2,2,2,2,2,2,2,2)*im;#Aup,Adown, Bup,Bdown
SS[2,1,1,2,1,2,2,1]=1/2;#spsm
SS[1,2,2,1,2,1,1,2]=1/2;#smmsp
SS[2,1,2,1,2,1,2,1]=1/4;#szsz
SS[2,1,1,2,2,1,1,2]=-1/4;#szsz
SS[1,2,2,1,1,2,2,1]=-1/4;#szsz
SS[1,2,1,2,1,2,1,2]=1/4;#szsz
SS=reshape(SS,4,4,4,4);
@tensor SS[:]:=SS[1,2,3,4]*Pm[-1,1]*Pm[-2,2]*Pm[-3,3]*Pm[-4,4];
SS_op_F=TensorMap(SS, V' ⊗ V' ← V' ⊗ V');


#Si dot Sj cross Sk
SS=zeros(2,2,2,2,2,2,2,2,2,2,2,2)*im;#Aup,Adown, Bup,Bdown, Cup,Cdown
SS[2,1,1,2,2,1, 2,1,2,1,1,2]=im/4;
SS[1,2,2,1,2,1, 2,1,2,1,1,2]=-im/4;

SS[1,2,2,1,2,1, 2,1,1,2,2,1]=im/4;
SS[2,1,2,1,1,2, 2,1,1,2,2,1]=-im/4;

SS[2,1,2,1,1,2, 1,2,2,1,2,1]=im/4;
SS[2,1,1,2,2,1, 1,2,2,1,2,1]=-im/4;

SS[1,2,2,1,1,2, 1,2,1,2,2,1]=im/4;
SS[2,1,1,2,1,2, 1,2,1,2,2,1]=-im/4;

SS[2,1,1,2,1,2, 1,2,2,1,1,2]=im/4;
SS[1,2,1,2,2,1, 1,2,2,1,1,2]=-im/4;

SS[1,2,1,2,2,1, 2,1,1,2,1,2]=im/4;
SS[1,2,2,1,1,2, 2,1,1,2,1,2]=-im/4;

SS=reshape(SS,4,4,4,4,4,4);
@tensor SS[:]:=SS[1,2,3,4,5,6]*Pm[-1,1]*Pm[-2,2]*Pm[-3,3]*Pm[-4,4]*Pm[-5,5]*Pm[-6,6];
Schiral_op_F=TensorMap(SS, V' ⊗ V' ⊗ V' ← V' ⊗ V' ⊗ V');



#gutzwiller projector
P=zeros(2,2,2)*im;
P[1,2,1]=1;
P[2,1,2]=1;
P=reshape(P,2,4);
@tensor P[:]:=P[-1,1]*Pm[-2,1];
Vspin=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
P_G=TensorMap(P, Vspin'  ←  V');

@tensor SS_op_S[:]:=P_G[-1,1]*P_G[-2,2]*SS_op_F[1,2,3,4]*P_G'[3,-3]*P_G'[4,-4];
SS_op_S=permute(SS_op_S,(1,2,),(3,4,))

@tensor Schiral_op_S[:]:=P_G[-1,1]*P_G[-2,2]*P_G[-3,3]*Schiral_op_F[1,2,3,4,5,6]*P_G'[4,-4]*P_G'[5,-5]*P_G'[6,-6];
Schiral_op_S=permute(Schiral_op_S,(1,2,3,),(4,5,6,))

save("swap_gate_Tensor_M2.jld2", "psi_G",psi_G, "bond",bond,"bond_gate",bond_gate, "SS_op_F",SS_op_F, "SS_op_S",SS_op_S, "P_G",P_G, "Schiral_op_F",Schiral_op_F, "Schiral_op_S",Schiral_op_S);



