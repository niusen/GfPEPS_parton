using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("double_layer_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")




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
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
#############################
println(space(A1,4))
V_odd,V_even=projector_virtual(space(A1,4))
#physical state only has even parity

l_Vodd=length(V_odd);
l_Veven=length(V_even);


AA1_Vodd=Vector(undef,l_Vodd);#upper layer, lower layer
AA1_Veven=Vector(undef,l_Veven);#upper layer, lower layer


AA3_Vodd=Vector(undef,l_Vodd);#upper layer, lower layer
AA3_Veven=Vector(undef,l_Veven);#upper layer, lower layer




for cc1=1:l_Vodd

        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc1][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_swap(Ap_temp,A_temp);
        @tensor AA_temp[:]:=AA_temp[-1,-2,-3,1]*U_U'[1,2,2];
        AA1_Vodd[cc1]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,-5]*V_odd[cc1][-2,1];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,-5]*V_odd[cc1]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_swap(Ap_temp,A_temp);
        @tensor AA_temp[:]:=AA_temp[-1,1,-2,-3]*U_D'[2,2,1];
        AA3_Vodd[cc1]=AA_temp;

end

for cc1=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc1][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_swap(Ap_temp,A_temp);
        @tensor AA_temp[:]:=AA_temp[-1,-2,-3,1]*U_U'[1,2,2];
        AA1_Veven[cc1]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,-5]*V_even[cc1][-2,1];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,-5]*V_even[cc1]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_swap(Ap_temp,A_temp);
        @tensor AA_temp[:]:=AA_temp[-1,1,-2,-3]*U_D'[2,2,1];
        AA3_Veven[cc1]=AA_temp;

end



AA2, U_L,U_D,U_R,U_U=build_double_layer_swap(deepcopy(A2'),deepcopy(A2));
AA2=AA2/norm(AA2);

gate_upper=parity_gate(U_L,2); @tensor gate_upper[:]:=gate_upper[2,1]*U_L[-1,1,3]*U_L'[2,3,-2];
gate_lower=parity_gate(U_L,3); @tensor gate_lower[:]:=gate_lower[2,1]*U_L[-1,3,1]*U_L'[3,2,-2];

global AA1_Vodd, AA1_Veven
global AA3_Vodd, AA3_Veven
global AA2
global gate_upper, gate_lower

function M_vr(l_Vodd,l_Veven,vr0)
    vr=deepcopy(vr0)*0;#L1,L2,L3,dummy
    #sign from U1+U1'
    #sign from P2*(R1+R1')+P3*(R1+R1'+R2+R2')
    #sign from U1*(L1+L2+L3)
    #sign from U1'*(L1'+L2'+L3')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd[cc1]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3_Vodd[cc2]);
            #gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3') 
            @tensor vr_temp[:]:=AA1_temp[-1,2,1]*AA2_temp[-2,4,3,2]*AA3_temp[-3,5,4]*vr_temp[1,3,5,-4];
            vr=vr+vr_temp;

        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd[cc1]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3_Veven[cc2]);
            #gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3')                    
            @tensor vr_temp[:]:=AA1_temp[-1,2,1]*AA2_temp[-2,4,3,2]*AA3_temp[-3,5,4]*vr_temp[1,3,5,-4];
            vr=vr+vr_temp;

        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Veven[cc1]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3_Vodd[cc2]);
            #gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3')                    
            @tensor vr_temp[:]:=AA1_temp[-1,2,1]*AA2_temp[-2,4,3,2]*AA3_temp[-3,5,4]*vr_temp[1,3,5,-4];
            vr=vr+vr_temp;

        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Veven[cc1]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3_Veven[cc2]);
            #gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            #No sign for U1*(L1+L2+L3) + U1'*(L1'+L2'+L3') 
            @tensor vr_temp[:]:=AA1_temp[-1,2,1]*AA2_temp[-2,4,3,2]*AA3_temp[-3,5,4]*vr_temp[1,3,5,-4];
            vr=vr+vr_temp;

        end
    end


    return vr;
end

v_init=TensorMap(randn, space(AA2,1)*space(AA2,1)*space(AA2,1),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1));
v_init=permute(v_init,(1,2,3,4,),());#L1,L2,L3,dummy
contraction_fun_R(x)=M_vr(l_Vodd,l_Veven,x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 3,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,dummy

println(eur)

v_init=TensorMap(randn, space(AA2,1)*space(AA2,1)*space(AA2,1),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1));
v_init=permute(v_init,(1,2,3,4,),());#L1,L2,L3,dummy
contraction_fun_R(x)=M_vr(l_Vodd,l_Veven,x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 3,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,dummy

println(eur)