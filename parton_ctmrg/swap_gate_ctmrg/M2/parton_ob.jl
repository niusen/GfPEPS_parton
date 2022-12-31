using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M2")

include("parton_CTMRG.jl")
include("parton_model.jl")
include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")



chi=10
tol=1e-6




CTM_ite_nums=500;
CTM_trun_tol=1e-10;


data=load("swap_gate_Tensor_M2.jld2")
psi_G=data["psi_G"];   #P1,P2,L,R,D,U
M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];
M6=psi_G[6];
M7=psi_G[7];
M8=psi_G[8];
M9=psi_G[9];
M10=psi_G[10];

U_phy1=unitary(fuse(space(M1,1)⊗space(M1,3)⊗space(M2,3)), space(M1,1)⊗space(M1,3)⊗space(M2,3));

@tensor A[:]:=M1[4,1,2]*M2[1,-2,3]*U_phy1[-1,4,2,3];
@tensor A[:]:=A[-1,1]*M3[1,-3,-2];
@tensor A[:]:=A[-1,-2,1]*M4[1,-4,-3];
@tensor A[:]:=A[-1,-2,-3,1]*M5[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M6[1,-6,-5];
@tensor A[:]:=A[-1,-2,-3,-4,-5,1]*M7[1,-7,-6];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1]*M8[1,-8,-7];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,1]*M9[1,-9,-8];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,-8,1]*M10[1,-10,-9];

U_phy_dummy=unitary(fuse(space(A,1)⊗space(A,10)), space(A,1)⊗space(A,10));#this doesn't do anything
@tensor A[:]:=A[1,-2,-3,-4,-5,-6,-7,-8,-9,2]*U_phy_dummy[-1,1,2];
# P,L,R,D,U


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-6,-7,1,2,3,4,-12,-13]*bond[-2,-8,1]*bond[-3,-9,2]*bond[-4,-10,3]*bond[-5,-11,4];

U_phy2=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)⊗space(A,4)⊗space(A,5)), space(A,1)⊗space(A,2)⊗space(A,3)⊗space(A,4)⊗space(A,5));
@tensor A[:]:=A[1,2,3,4,5,-2,-3,-4,-5,-6,-7,-8,-9]*U_phy2[-1,1,2,3,4,5];
#P,L,R,D,U





#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5,-6,-7,-8,-9]*special_gate[-4,1];
@tensor A[:]:=A[-1,-2,-3,-4,1,-6,-7,-8,-9]*special_gate[-5,1];
@tensor A[:]:=A[-1,-2,-3,-4,-5,1,-7,-8,-9]*special_gate[-6,1];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1,-8,-9]*special_gate[-7,1];

gate=swap_gate(A,4,5);@tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2];  
gate=swap_gate(A,6,7);@tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2];  



#group virtual legs on the same legs
U1=unitary(fuse(space(A,2)⊗space(A,3)),space(A,2)⊗space(A,3)); 
U2=unitary(fuse(space(A,8)⊗space(A,9)),space(A,8)⊗space(A,9));
@tensor A[:]:=A[-1,1,2,-3,-4,-5,-6,-7,-8]*U1[-2,1,2];
@tensor A[:]:=A[-1,-2,1,2,-4,-5,-6,-7]*U2'[1,2,-3];
@tensor A[:]:=A[-1,-2,-3,1,2,-5,-6]*U1'[1,2,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1,2]*U2[-5,1,2];



A1=deepcopy(A);


gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
A=permute(A,(1,3,2,4,5,));#L,U,P,R,D

#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


#convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));









A_fused=A;


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true);
@show varinfo()
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);



display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))




#NANB
W1=zeros(2,1);W1[1,1]=1;
W2=zeros(2,1);W2[2,1]=1;
W1_set=create_mpo(W1);
W2_set=create_mpo(W2);
@tensor W[:]:=W1_set[1][-1,1,2,-5]*W1_set[2][2,4,-3,-6]*W2_set[1][-2,-7,3,1]*W2_set[2][3,-8,-4,4];
U=unitary(fuse(space(W,1)⊗space(W,2)⊗space(W,3)⊗space(W,4)), space(W,1)⊗space(W,2)⊗space(W,3)⊗space(W,4));
@tensor W[:]:=W[1,2,3,4,-2,-3,-4,-5]*U[-1,1,2,3,4];
@tensor NANB[:]:=W[1,-1,-2,2,3]*W'[1,-3,-4,2,3];
NANB=permute(NANB,(1,2,),(3,4,))

Ident=unitary(space(NANB,1)⊗space(NANB,1),space(NANB,1)⊗space(NANB,1));


#NA
W1=zeros(2,1);W1[1,1]=1;
W1_set=create_mpo(W1);
@tensor W[:]:=W1_set[1][-1,-5,1,-3]*W1_set[2][1,-6,-2,-4];
U=unitary(fuse(space(W,1)⊗space(W,2)), space(W,1)⊗space(W,2));
@tensor W[:]:=W[1,2,-2,-3,-4,-5]*U[-1,1,2];
@tensor NA[:]:=W[1,-1,-2,2,3]*W'[1,-3,-4,2,3];
NA=permute(NA,(1,2,),(3,4,))

#NB
W1=zeros(2,1);W1[2,1]=1;
W1_set=create_mpo(W1);
@tensor W[:]:=W1_set[1][-1,-5,1,-3]*W1_set[2][1,-6,-2,-4];
U=unitary(fuse(space(W,1)⊗space(W,2)), space(W,1)⊗space(W,2));
@tensor W[:]:=W[1,2,-2,-3,-4,-5]*U[-1,1,2];
@tensor NB[:]:=W[1,-1,-2,2,3]*W'[1,-3,-4,2,3];
NB=permute(NB,(1,2,),(3,4,))


#CAdag_CB_onsite
W1=zeros(2,1);W1[1,1]=1;
W2=zeros(2,1);W2[2,1]=1;
W1_set=create_mpo(W1);
W2_set=create_mpo(W2);
for cc=1:length(W2_set)
    W2_set[cc]=permute(W2_set[cc],(1,2,3,4,));
    W2_set[cc]=W2_set[cc]';
end
@tensor CAdag_CB_onsite[:]:=W1_set[1][5,1,2,-1]*W1_set[2][2,4,6,-2]*W2_set[1][5,1,3,-3]*W2_set[2][3,4,6,-4];
CAdag_CB_onsite=permute(CAdag_CB_onsite,(1,2,),(3,4,))


@tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
@tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];




@tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
@tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,5,6,1,2]*U_phy2'[4,1,2,5,6,-2];


@tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
@tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

@tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
@tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

@tensor CAdag_CB_onsite[:]:=CAdag_CB_onsite[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
@tensor CAdag_CB_onsite[:]:=CAdag_CB_onsite[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];


O1=NANB;
O2=Ident;
direction="x";
is_odd=false;
ob_NANB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
println("NANB= "*string(ob_NANB))

O1=NA;
O2=Ident;
direction="x";
is_odd=false;
ob_NA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
println("NA= "*string(ob_NA))

O1=NB;
O2=Ident;
direction="x";
is_odd=false;
ob_NB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
println("NB= "*string(ob_NB))

O1=CAdag_CB_onsite;
O2=Ident;
direction="x";
is_odd=false;
ob_CAdag_CB_onsite=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
println("CAdag_CB_onsite= "*string(ob_CAdag_CB_onsite))





# Ident, NA, NB, NANB, CAdag, CA, CBdag, CB=Hamiltonians(U_phy1,U_phy2)

# O1=NA;
# O2=Ident;
# direction="x";
# is_odd=false;
# NA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# O1=NB;
# O2=Ident;
# direction="x";
# is_odd=false;
# NB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# O1=NANB;
# O2=Ident;
# direction="x";
# is_odd=false;
# NANB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# @tensor O1[:]:=CAdag[1,-1,2]*CB[1,2,-2];
# O2=Ident;
# direction="x";
# is_odd=false;
# CAdagCB_onsite=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)





# O1=CAdag;
# O2=CA;
# direction="x";
# is_odd=true;
# CAdag_CA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# O1=CAdag;
# O2=CB;
# direction="x";
# is_odd=true;
# CAdag_CB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# O1=CBdag;
# O2=CA;
# direction="x";
# is_odd=true;
# CBdag_CA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# O1=CBdag;
# O2=CB;
# direction="x";
# is_odd=true;
# CBdag_CB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

# println("NA=   "*string(NA))
# println("NB=   "*string(NB))
# println("NANB=   "*string(NANB))
# println("CAdagCB_onsite=   "*string(CAdagCB_onsite))

# println("CAdag_CA=   "*string(CAdag_CA))
# println("CAdag_CB=   "*string(CAdag_CB))
# println("CBdag_CA=   "*string(CBdag_CA))
# println("CBdag_CB=   "*string(CBdag_CB))





