using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M2")

include("parton_CTMRG.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\correl_funs.jl")
include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")

swap_gate_double_layer=true;
M=2;#number of virtual mode
distance=10;
chi=20
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
M7=psi_G[7];
M8=psi_G[8];
M9=psi_G[9];
M10=psi_G[10];

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
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,swap_gate_double_layer);

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

if Guztwiller
    if swap_gate_double_layer
        AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
        AA_SAL=build_double_layer_swap_op(A_fused,SA_left,true);
        AA_SBL=build_double_layer_swap_op(A_fused,SB_left,true);
        AA_SAR=build_double_layer_swap_op(A_fused,SA_right,true);
        AA_SBR=build_double_layer_swap_op(A_fused,SB_right,true);
    else 
        AA_SS=build_double_layer_NoSwap_op(A_fused,SS_cell,false);
        AA_SAL=build_double_layer_NoSwap_op(A_fused,SA_left,true);
        AA_SBL=build_double_layer_NoSwap_op(A_fused,SB_left,true);
        AA_SAR=build_double_layer_NoSwap_op(A_fused,SA_right,true);
        AA_SBR=build_double_layer_NoSwap_op(A_fused,SB_right,true);
    end
else
    AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
    AA_SAL=build_double_layer_swap_op(A_fused,SA_left,true);
    AA_SBL=build_double_layer_swap_op(A_fused,SB_left,true);
    AA_SAR=build_double_layer_swap_op(A_fused,SA_right,true);
    AA_SBR=build_double_layer_swap_op(A_fused,SB_right,true);
end

println("construct double layer tensor with operator");flush(stdout);
AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
AA_SAL=build_double_layer_swap_op(A_fused,SA_left,true);
AA_SBL=build_double_layer_swap_op(A_fused,SB_left,true);
AA_SAR=build_double_layer_swap_op(A_fused,SA_right,true);
AA_SBR=build_double_layer_swap_op(A_fused,SB_right,true);

@show varinfo()

println("Calculate correlations:");flush(stdout);
cal_correl(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)


