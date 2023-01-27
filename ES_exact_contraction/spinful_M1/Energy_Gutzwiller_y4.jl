using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("swap_funs.jl")
include("fermi_permute.jl")

include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")
include("projected_energy.jl")


sublattice_order="RL";
M=1;
Guztwiller=true;#add projector


data=load("swap_gate_Tensor_M"*string(M)*".jld2");
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
    Schiral_op=data["Schiral_op_S"];
else
    SS_op=data["SS_op_F"];
    Schiral_op=data["Schiral_op_F"];
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


A_origin=deepcopy(A);




y_anti_pbc=true;
boundary_phase_y=0.0;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi/4*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
A4=deepcopy(A);
#############################

println(space(A1,4))



A1=A1/norm(A1);
A2=A2/norm(A2);
A3=A3/norm(A3);
A4=A4/norm(A4);









U_DD=unitary(fuse(space(A2,1)⊗space(A2,1)), space(A2,1)⊗space(A2,1));
U_DD_DD=unitary(fuse(space(U_DD,1)⊗space(U_DD,1)), space(U_DD,1)⊗space(U_DD,1));
U_PP=unitary(fuse(space(A2,5)⊗space(A2,5)), space(A2,5)⊗space(A2,5));
U_PPPP=unitary(fuse(space(U_PP,1)⊗space(U_PP,1)), space(U_PP,1)⊗space(U_PP,1));


gate_U=parity_gate(A1,4); 
@tensor A1[:]:=A1[-1,-2,-3,1,-5]*gate_U[-4,1];


@tensor A1A2[:]:=A1[1,3,4,-4,6]*A2[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
@tensor A3A4[:]:=A3[1,3,4,-4,6]*A4[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
@tensor A1A2A3A4[:]:=A1A2[-1,1,-4,-6,2]*A3A4[-2,-3,-5,1,3]*U_PPPP[-7,2,3];
gate=swap_gate(A1A2A3A4,1,6); @tensor A1A2A3A4[:]:=A1A2A3A4[1,-2,-3,-4,-5,2,-7]*gate[-1,-6,1,2];
gate=swap_gate(A1A2A3A4,2,6); @tensor A1A2A3A4[:]:=A1A2A3A4[-1,1,-3,-4,-5,2,-7]*gate[-2,-6,1,2];

@tensor A1A2A3A4[:]:=A1A2A3A4[1,2,3,4,5,3,-3]*U_DD_DD[-1,1,2]*U_DD_DD'[4,5,-2];




global A1A2A3A4


function M_vr(vr0)
    vr=deepcopy(vr0)*0;#L1'L2',L3'L4',L1L2,L3L4
    for cp=1:l_P
        @tensor vr_temp[:]:=vr0[3,1]*A1A2A3A4'[-1,3,2]*A1A2A3A4[-2,1,2];
        vr=vr+vr_temp;
    end
    println("finished one Mv operation");flush(stdout);
    return vr;
end

function vl_M(vl0)
    vl=deepcopy(vl0)*0;#R1'R2',R3'R4',R1R2,R3R4
    for cp=1:l_P
        @tensor vl_temp[:]:=vl0[3,1]*A1A2A3A4'[3,-1,2]*A1A2A3A4[1,-2,2];
        vl=vl+vl_temp;
    end
    println("finished one Mv operation");flush(stdout);
    return vl;
end
v_init=TensorMap(randn, space(U_DD_DD,1)',space(U_DD_DD,1)');
v_init=permute(v_init,(1,2,),());#L1'L2',L3'L4',L1L2,L3L4
contraction_fun_R(x)=M_vr(x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 3,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1'L2',L3'L4',L1L2,L3L4

println(eur)


v_init=TensorMap(randn, space(U_DD_DD,1),space(U_DD_DD,1));
v_init=permute(v_init,(1,2,),());#R1'R2',R3'R4',R1R2,R3R4
contraction_fun_L(x)=vl_M(x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 3,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#R1'R2',R3'R4',R1R2,R3R4

println(eul)



##################




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



@tensor op[:]:=U_PPPP[-1,8,9]*U_PP[8,2,3]*U_PP[9,4,5]*SS_cell[2,1]*U_PP'[1,3,6]*U_PP'[4,5,7]*U_PPPP'[6,7,-2];
@tensor ov[:]:=VL[3,2]*A1A2A3A4'[3,5,4]*op[4,1]*A1A2A3A4[2,6,1]*VR[5,6];
ov=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
@tensor Norm[:]:=VL[3,2]*A1A2A3A4'[3,5,4]*A1A2A3A4[2,6,4]*VR[5,6];
Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println(ov/Norm)


@tensor op[:]:=U_PPPP[-1,10,11]*U_PP[10,2,3]*U_PP[11,6,7]*S_L_a[2,4,1]*S_L_b[3,5,1]*U_PP'[4,5,8]*U_PP'[6,7,9]*U_PPPP'[8,9,-2];
@tensor ov[:]:=VL[3,2]*A1A2A3A4'[3,5,4]*op[4,1]*A1A2A3A4[2,6,1]*VR[5,6];
ov=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
@tensor Norm[:]:=VL[3,2]*A1A2A3A4'[3,5,4]*A1A2A3A4[2,6,4]*VR[5,6];
Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println(ov/Norm)

@tensor op[:]:=U_PPPP[-1,10,11]*U_PP[10,2,3]*U_PP[11,6,7]*S_R_a[2,4,1]*S_R_b[3,5,1]*U_PP'[4,5,8]*U_PP'[6,7,9]*U_PPPP'[8,9,-2];
@tensor ov[:]:=VL[3,2]*A1A2A3A4'[3,5,4]*op[4,1]*A1A2A3A4[2,6,1]*VR[5,6];
ov=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
@tensor Norm[:]:=VL[3,2]*A1A2A3A4'[3,5,4]*A1A2A3A4[2,6,4]*VR[5,6];
Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println(ov/Norm)









