using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg\\mpo_swap_construction\\M2")

include("GfPEPS_CTMRG.jl")
include("GfPEPS_model.jl")
include("swap_funs.jl")



chi=30
tol=1e-6




CTM_ite_nums=500;
CTM_trun_tol=1e-14;



data=load("swap_gate_Tensor_M2.jld2")
psi_G=data["psi_G"];   #P,L,R,D,U
M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];



@tensor A[:]:=M1[-1,1,-2]*M2[1,-4,-3]
@tensor A[:]:=A[-1,-2,-3,1]*M3[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M4[1,-6,-5];
@tensor A[:]:=A[-1,-2,-3,-4,-5,1]*M5[1,-7,-6];

U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,7)), space(A,1)⊗space(A,2)⊗space(A,7));


@tensor A[:]:=A[1,2,-2,-3,-4,-5,3]*U_phy1[-1,1,2,3];
# P,L,R,D,U


A1=deepcopy(A);


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U

A11=deepcopy(A);

#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,3);
@tensor A[:]:=A[-1,-2,1,-4,-5]*special_gate[-3,1];
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5]*special_gate[-4,1];




A111=deepcopy(A);


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
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);



display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))

Ident, NA, NB, NANB, CAdag, CA, CBdag, CB=Hamiltonians(U_phy1,U_phy2)

O1=NA;
O2=Ident;
direction="x";
is_odd=false;
NA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=NB;
O2=Ident;
direction="x";
is_odd=false;
NB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=NANB;
O2=Ident;
direction="x";
is_odd=false;
NANB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)


@tensor O1[:]:=CAdag[1,-1,2]*CB[1,2,-2];
O2=Ident;
direction="x";
is_odd=false;
CAdagCB_onsite=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)





O1=CAdag;
O2=CA;
direction="x";
is_odd=true;
CAdag_CA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=CAdag;
O2=CB;
direction="x";
is_odd=true;
CAdag_CB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=CBdag;
O2=CA;
direction="x";
is_odd=true;
CBdag_CA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=CBdag;
O2=CB;
direction="x";
is_odd=true;
CBdag_CB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

println("NA=   "*string(NA))
println("NB=   "*string(NB))
println("NANB=   "*string(NANB))
println("CAdagCB_onsite=   "*string(CAdagCB_onsite))

println("CAdag_CA=   "*string(CAdag_CA))
println("CAdag_CB=   "*string(CAdag_CB))
println("CBdag_CA=   "*string(CBdag_CA))
println("CBdag_CB=   "*string(CBdag_CB))





