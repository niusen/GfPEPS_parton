using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("/users/p1231/niu/Code/Julia_codes/Tensor_network/GfPEPS_parton/test_M2_projected/CTMRG_funs.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/GfPEPS_parton/test_M2_projected/projected_energy.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/GfPEPS_parton/test_M2_projected/swap_funs.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/GfPEPS_parton/test_M2_projected/mpo_mps_funs.jl")

M=2;
chi=80;
sublattice_order="RL";
tol=1e-6
CTM_ite_nums=500;
CTM_trun_tol=1e-10;
println("chi="*string(chi));flush(stdout);


Guztwiller=true;#add projector



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


A_origin=deepcopy(A);





#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));

#############################


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,"PBC",true);
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
if sublattice_order=="LR"
    @tensor S_L_a[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_R_a[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor S_L_b[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_R_b[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
elseif sublattice_order=="RL"
    @tensor S_R_a[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_L_a[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor S_R_b[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_L_b[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
end

if M==1        
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_a[:]:=S_L_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_a[:]:=S_R_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_b[:]:=S_L_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_b[:]:=S_R_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
elseif M==2
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_L_a[:]:=S_L_a[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_R_a[:]:=S_R_a[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_L_b[:]:=S_L_b[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_R_b[:]:=S_R_b[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
end



####################################
#chiral operator act on a single site: Si Sj Sk
um,sm,vm=tsvd(Schiral_op,(1,4,),(2,3,5,6,));
vm=sm*vm;
Si=permute(um,(1,2,3,));#P,P',D1
um,sm,vm=tsvd(vm,(1,2,4,),(3,5,));
vm=sm*vm;
Sj=permute(um,(2,3,1,4,));#P,P', D1,D2
Sk=permute(vm,(2,3,1,))#P,P',D2
@tensor SiSj[:]:=Si[-1,-3,1]*Sj[-2,-4,1,-5]; 
@tensor SjSk[:]:=Sj[-1,-3,-5,1]*Sk[-2,-4,1]; 
#@tensor aa[:]:=Si[-1,-4,1]*Sj[-2,-5,1,2]*Sk[-3,-6,2];
U_Schiral=unitary(fuse(space(Sj,3)⊗space(Sj,4)), space(Sj,3)⊗space(Sj,4));
@tensor Sj[:]:=Sj[-1,-2,1,2]*U_Schiral[-3,1,2];#combine two extra indices of Sj


if sublattice_order=="LR"
    @tensor Si_left[:]:=Si[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Si_right[:]:=Si[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sj_left[:]:=Sj[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sj_right[:]:=Sj[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sk_left[:]:=Sk[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sk_right[:]:=Sk[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];

    @tensor SiSj_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SjSi_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
    @tensor SjSk_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SkSj_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
elseif sublattice_order=="RL"
    @tensor Si_right[:]:=Si[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Si_left[:]:=Si[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sj_right[:]:=Sj[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sj_left[:]:=Sj[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sk_right[:]:=Sk[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sk_left[:]:=Sk[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];

    @tensor SjSi_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SiSj_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
    @tensor SkSj_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SjSk_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
end

if M==1        
    @tensor Si_left[:]:=Si_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Si_right[:]:=Si_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sj_left[:]:=Sj_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sj_right[:]:=Sj_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sk_left[:]:=Sk_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sk_right[:]:=Sk_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SiSj_op[:]:=SiSj_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SjSi_op[:]:=SjSi_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SjSk_op[:]:=SjSk_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SkSj_op[:]:=SkSj_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
elseif M==2
    @tensor Si_left[:]:=Si_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Si_right[:]:=Si_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sj_left[:]:=Sj_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sj_right[:]:=Sj_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sk_left[:]:=Sk_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sk_right[:]:=Sk_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SiSj_op[:]:=SiSj_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SjSi_op[:]:=SjSi_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SjSk_op[:]:=SjSk_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SkSj_op[:]:=SkSj_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
end
################################################


println("construct double layer tensor with operator");flush(stdout);
AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
AA_SLa=build_double_layer_swap_op(A_fused,S_L_a,true);
AA_SRa=build_double_layer_swap_op(A_fused,S_R_a,true);
AA_SLb=build_double_layer_swap_op(A_fused,S_L_b,true);
AA_SRb=build_double_layer_swap_op(A_fused,S_R_b,true);

AA_SiL=build_double_layer_swap_op(A_fused,Si_left,true);
AA_SiR=build_double_layer_swap_op(A_fused,Si_right,true);
AA_SjL=build_double_layer_swap_op(A_fused,Sj_left,true);
AA_SjR=build_double_layer_swap_op(A_fused,Sj_right,true);
AA_SkL=build_double_layer_swap_op(A_fused,Sk_left,true);
AA_SkR=build_double_layer_swap_op(A_fused,Sk_right,true);
AA_SiSj=build_double_layer_swap_op(A_fused,SiSj_op,true);
AA_SjSi=build_double_layer_swap_op(A_fused,SjSi_op,true);
AA_SjSk=build_double_layer_swap_op(A_fused,SjSk_op,true);
AA_SkSj=build_double_layer_swap_op(A_fused,SkSj_op,true);


println("Calculate energy terms:");flush(stdout);


Norm_1=ob_1site_closed(CTM,AA_fused)
Norm_2x=norm_2sites_x(CTM,AA_fused)
Norm_2y=norm_2sites_y(CTM,AA_fused)
Norm_4=norm_2x2(CTM,AA_fused);

#J1 term
E_1_a=ob_1site_closed(CTM,AA_SS)/Norm_1
E_1_b=ob_2sites_x(CTM,AA_SRa,AA_SLb)/Norm_2x
E_1_c=ob_2sites_y(CTM,AA_SLa,AA_SLb)/Norm_2y
E_1_d=ob_2sites_y(CTM,AA_SRa,AA_SRb)/Norm_2y

#J2 term
E_2_a=ob_2sites_y(CTM,AA_SLa,AA_SRb)/Norm_2y
E_2_b=ob_2sites_y(CTM,AA_SRa,AA_SLb)/Norm_2y
E_2_c=ob_LU_RD(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
E_2_d=ob_RU_LD(CTM,AA_fused,AA_SLa,AA_SRb)/Norm_4



#chiral term
E_C_a=ob_2sites_y(CTM,AA_SiSj,AA_SkR)/Norm_2y
E_C_b=ob_2sites_y(CTM,AA_SiR,AA_SkSj)/Norm_2y
E_C_c=ob_2sites_y(CTM,AA_SjSk,AA_SiL)/Norm_2y
E_C_d=ob_2sites_y(CTM,AA_SiSj,AA_SkL)/Norm_2y

E_C_e=ob_LD_LU_RU(CTM,AA_fused,AA_SiR,AA_SjR,AA_SkL,U_Schiral)/Norm_4
E_C_f=ob_LU_RU_RD(CTM,AA_fused,AA_SiR,AA_SjL,AA_SkL,U_Schiral)/Norm_4
E_C_g=ob_RU_RD_LD(CTM,AA_fused,AA_SiL,AA_SjL,AA_SkR,U_Schiral)/Norm_4
E_C_h=ob_RD_LD_LU(CTM,AA_fused,AA_SiL,AA_SjR,AA_SkR,U_Schiral)/Norm_4

println("J1 terms:")
println(E_1_a)
println(E_1_b)
println(E_1_c)
println(E_1_d)
println("J2 terms:")
println(E_2_a)
println(E_2_b)
println(E_2_c)
println(E_2_d)
println("Jchi terms:")
println(E_C_a)
println(E_C_b)
println(E_C_c)
println(E_C_d)
println(E_C_e)
println(E_C_f)
println(E_C_g)
println(E_C_h)



