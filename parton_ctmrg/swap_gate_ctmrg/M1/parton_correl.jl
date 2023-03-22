using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M1")

include("parton_CTMRG.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\correl_funs.jl")
include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")


M=1;#number of virtual mode
distance=40;
chi=120
tol=1e-6
Guztwiller=false;#add projector



CTM_ite_nums=500;
CTM_trun_tol=1e-10;


data=load("swap_gate_Tensor_M"*string(M)*".jld2")

P_G=data["P_G"];

psi_G=data["psi_G"];   #P1,P2,L,R,D,U
M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];
M6=psi_G[6];

if Guztwiller
    @tensor M1[:]:=M1[-1,-2,1]*P_G[-3,1];
    @tensor M2[:]:=M2[-1,-2,1]*P_G[-3,1];
    SS_op=data["SS_op_S"];
else
    SS_op=data["SS_op_F"];
end


U_phy1=unitary(fuse(space(M1,1)⊗space(M1,3)⊗space(M2,3)), space(M1,1)⊗space(M1,3)⊗space(M2,3));

@tensor A[:]:=M1[4,1,2]*M2[1,-2,3]*U_phy1[-1,4,2,3];
@tensor A[:]:=A[-1,1]*M3[1,-3,-2];
@tensor A[:]:=A[-1,-2,1]*M4[1,-4,-3];
@tensor A[:]:=A[-1,-2,-3,1]*M5[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M6[1,-6,-5];

U_phy2=unitary(fuse(space(A,1)⊗space(A,6)), space(A,1)⊗space(A,6));
@tensor A[:]:=A[1,-2,-3,-4,-5,2]*U_phy2[-1,1,2];
# P,L,R,D,U


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U





#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,3);
@tensor A[:]:=A[-1,-2,1,-4,-5]*special_gate[-3,1];
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5]*special_gate[-4,1];



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


println("construct physical operators");flush(stdout);
#spin-spin operator act on a single site
um,sm,vm=tsvd(permute(SS_op,(1,3,),(2,4,)));
vm=sm*vm;vm=permute(vm,(2,3,),(1,));

@tensor SS_cell[:]:=SS_op[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];#spin-spin operator inside a unitcell
@tensor SA_left[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
@tensor SB_left[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
@tensor SA_right[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
@tensor SB_right[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];

if M==1        
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SA_left[:]:=SA_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SB_left[:]:=SB_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SA_right[:]:=SA_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SB_right[:]:=SB_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
elseif M==2
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SA_left[:]:=SA_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SB_left[:]:=SB_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SA_right[:]:=SA_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SB_right[:]:=SB_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
end

println("construct double layer tensor with operator");flush(stdout);
if Guztwiller
    AA_SS=build_double_layer_NoSwap_op(A_fused,SS_cell,false);
    AA_SAL=build_double_layer_NoSwap_op(A_fused,SA_left,true);
    AA_SBL=build_double_layer_NoSwap_op(A_fused,SB_left,true);
    AA_SAR=build_double_layer_NoSwap_op(A_fused,SA_right,true);
    AA_SBR=build_double_layer_NoSwap_op(A_fused,SB_right,true);
else
    AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
    AA_SAL=build_double_layer_swap_op(A_fused,SA_left,true);
    AA_SBL=build_double_layer_swap_op(A_fused,SB_left,true);
    AA_SAR=build_double_layer_swap_op(A_fused,SA_right,true);
    AA_SBR=build_double_layer_swap_op(A_fused,SB_right,true);
end


println("Calculate correlations:");flush(stdout);
cal_correl(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)





