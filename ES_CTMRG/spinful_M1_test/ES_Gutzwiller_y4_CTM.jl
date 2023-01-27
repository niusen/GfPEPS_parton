using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\fermi_permute.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\double_layer_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")
include("CTMRG_funs_test.jl")


chi=80
tol=1e-6
CTM_ite_nums=500;
CTM_trun_tol=1e-10;


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


A_origin=deepcopy(A);




y_anti_pbc=true;
boundary_phase_y=0;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi/3*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));

#############################


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,"PBC",true);
global Tleft_gate,Tright_gate
Tleft_gate=deepcopy(CTM["Tset"][4]);
Tright_gate=deepcopy(CTM["Tset"][2]);

@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);

display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))









# v_init=TensorMap(randn, space(AA2,1)*space(AA2,1)*space(AA2,1),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1));
# v_init=permute(v_init,(1,2,3,4,),());#L1,L2,L3,dummy
# contraction_fun_R(x)=M_vr(l_Vodd,l_Veven,x,Sect);
# @time eur,evr=eigsolve(contraction_fun_R, v_init, 3,:LM,Arnoldi(krylovdim=10));
# VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,dummy


# v_init=TensorMap(randn, space(AA2,3)*space(AA2,3)*space(AA2,3),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1)');
# v_init=permute(v_init,(4,1,2,3,),());#dummy,R1,R2,R3
# contraction_fun_L(x)=vl_M(l_Vodd,l_Veven,x,Sect);
# @time eul,evl=eigsolve(contraction_fun_L, v_init, 3,:LM,Arnoldi(krylovdim=10));
# VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3

Tleft=CTM["Tset"][4];
Tright=CTM["Tset"][2];

################################
Sect="even";




@tensor VL[:]:=Tleft_gate[2,-1,1]*Tleft[3,-2,2]*Tleft[4,-3,3]*Tleft[1,-4,4];
@tensor VR[:]:=Tright_gate[1,-1,2]*Tright[2,-2,3]*Tright[3,-3,4]*Tright[4,-4,1];




@tensor VL[:]:=VL[1,2,3,4]*U_L[1,-1,-2]*U_L[2,-3,-4]*U_L[3,-5,-6]*U_L[4,-7,-8];#R1',R1,R2',R2,R3',R3
VL=permute(VL,(1,3,5,7,2,4,6,8,));#R1',R2',R3', R1,R2,R3


@tensor VR[:]:=VR[1,2,3,4]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4];#L1',L1,L2',L2,L3',L3
VR=permute(VR,(2,4,6,8,1,3,5,7,));#L1,L2,L3,L1',L2',L3'




@tensor H[:]:=VL[-1,-2,-3,-4,1,2,3,4]*VR[1,2,3,4,-5,-6,-7,-8];#R1',R2',R3' ,L1',L2',L3'



eu,ev=eig(H,(1,2,3,4,),(5,6,7,8,))
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)

println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,5);#L2',L1',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,5);#L2',L3',L1',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,5);#L2',L3',L4',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,-1]*ev[1,2,3,4,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;

order=findall(x -> mod(x,1) == 0, Spin)
k_phase=diag(k_phase);
eu=eu[order];
k_phase=k_phase[order];
Qn=Qn[order];
Spin=Spin[order]


order=sortperm(abs.(eu));
eu_set1=eu[order];
k_phase_set1=k_phase[order];
Qn_set1=Qn[order];
Spin_set1=Spin[order]

##########################################

Sect="odd";




@tensor VL[:]:=Tleft[2,-1,1]*Tleft[3,-2,2]*Tleft[4,-3,3]*Tleft[1,-4,4];
@tensor VR[:]:=Tright[1,-1,2]*Tright[2,-2,3]*Tright[3,-3,4]*Tright[4,-4,1];




@tensor VL[:]:=VL[1,2,3,4]*U_L[1,-1,-2]*U_L[2,-3,-4]*U_L[3,-5,-6]*U_L[4,-7,-8];#R1',R1,R2',R2,R3',R3
VL=permute(VL,(1,3,5,7,2,4,6,8,));#R1',R2',R3', R1,R2,R3


@tensor VR[:]:=VR[1,2,3,4]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4];#L1',L1,L2',L2,L3',L3
VR=permute(VR,(2,4,6,8,1,3,5,7,));#L1,L2,L3,L1',L2',L3'




@tensor H[:]:=VL[-1,-2,-3,-4,1,2,3,4]*VR[1,2,3,4,-5,-6,-7,-8];#R1',R2',R3' ,L1',L2',L3'



eu,ev=eig(H,(1,2,3,4,),(5,6,7,8,))
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)

println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,5);#L2',L1',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,5);#L2',L3',L1',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,5);#L2',L3',L4',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,-1]*ev[1,2,3,4,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;

order=findall(x -> mod(x,1) == 0.5, Spin)
k_phase=diag(k_phase);
eu=eu[order];
k_phase=k_phase[order];
Qn=Qn[order];
Spin=Spin[order]

order=sortperm(abs.(eu));
eu_set2=eu[order];
k_phase_set2=k_phase[order];
Qn_set2=Qn[order];
Spin_set2=Spin[order]

######################################
eu=vcat(eu_set1,eu_set2);
k_phase=vcat(k_phase_set1,k_phase_set2);
Qn=vcat(Qn_set1,Qn_set2);
Spin=vcat(Spin_set1,Spin_set2);

matwrite("ES_CTM_Gutzwiller_M1_Nv4"*"_chi"*string(chi)*".mat", Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Qn"=>Qn,
    "Spin"=>Spin
); compress = false)






