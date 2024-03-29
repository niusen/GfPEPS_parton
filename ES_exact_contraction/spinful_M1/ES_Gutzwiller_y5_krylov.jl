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
boundary_phase_y=0.0;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi/5*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
A4=deepcopy(A);
A5=deepcopy(A);
#############################

println(space(A1,4))
V_odd,V_even=projector_virtual(space(A1,4))
#physical state only has even parity


l_Vodd=length(V_odd);
l_Veven=length(V_even);


AA1_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA1_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA1_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA1_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer

AA5_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA5_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA5_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA5_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer



for cc1=1:l_Vodd
    for cc2=1:l_Vodd
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Vodd_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A5'[-1,1,-3,-4,-5]*V_odd[cc1][-2,1];
        @tensor A_temp[:]:=A5[-1,1,-3,-4,-5]*V_odd[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA5_Vodd_Vodd[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Vodd
    for cc2=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Vodd_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A5'[-1,1,-3,-4,-5]*V_odd[cc1][-2,1];
        @tensor A_temp[:]:=A5[-1,1,-3,-4,-5]*V_even[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA5_Vodd_Veven[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Veven
    for cc2=1:l_Vodd
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Veven_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A5'[-1,1,-3,-4,-5]*V_even[cc1][-2,1];
        @tensor A_temp[:]:=A5[-1,1,-3,-4,-5]*V_odd[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA5_Veven_Vodd[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Veven
    for cc2=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Veven_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A5'[-1,1,-3,-4,-5]*V_even[cc1][-2,1];
        @tensor A_temp[:]:=A5[-1,1,-3,-4,-5]*V_even[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA5_Veven_Veven[cc1,cc2]=AA_temp;

    end
end


AA2, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A2'),deepcopy(A2));
AA3, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A3'),deepcopy(A3));
AA4, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A4'),deepcopy(A4));

AA2=AA2/norm(AA2);
AA3=AA3/norm(AA3);
AA4=AA4/norm(AA4);


gate_upper=parity_gate(U_L,2); @tensor gate_upper[:]:=gate_upper[2,1]*U_L[-1,1,3]*U_L'[2,3,-2];
gate_lower=parity_gate(U_L,3); @tensor gate_lower[:]:=gate_lower[2,1]*U_L[-1,3,1]*U_L'[3,2,-2];

global AA1_Vodd_Vodd, AA1_Vodd_Veven, AA1_Veven_Vodd, AA1_Veven_Veven
global AA5_Vodd_Vodd, AA5_Vodd_Veven, AA5_Veven_Vodd, AA5_Veven_Veven
global AA2,AA3,AA4
global gate_upper, gate_lower

function M_vr(l_Vodd,l_Veven,vr0)
    vr=deepcopy(vr0)*0;#L1,L2,L3,L4,dummy
    #sign from U1+U1'
    #sign from U1*(L1+L2+L3+L4)
    #sign from U1'*(L1'+L2'+L3'+L4')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Vodd_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,10]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*vr_temp[1,3,5,7,9,-6];
            vr=vr+vr_temp;

        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Vodd_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                  
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,10]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*vr_temp[1,3,5,7,9,-6];
            vr=vr+vr_temp;

        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Veven_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                    
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,10]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*vr_temp[1,3,5,7,9,-6];
            vr=vr+vr_temp;

        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Veven_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            #No sign for U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,10]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*vr_temp[1,3,5,7,9,-6];
            vr=vr+vr_temp;

        end
    end


    return vr;
end

function vl_M(l_Vodd,l_Veven,vl0)
    vl=deepcopy(vl0)*0;#dummy,R1,R2,R3
    #sign from U1+U1'
    #sign from P2*(R1+R1')+P3*(R1+R1'+R2+R2')
    #sign from U1*(L1+L2+L3)
    #sign from U1'*(L1'+L2'+L3')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd

            vl_temp=deepcopy(vl0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Vodd_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,10]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*vl_temp[-1,1,3,5,7,9];
            vl=vl+vl_temp;

        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven

            vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
            AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Vodd_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                    
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,10]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*vl_temp[-1,1,3,5,7,9];
            vl=vl+vl_temp;

        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd

            vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
            AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Veven_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                  
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,10]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*vl_temp[-1,1,3,5,7,9];
            vl=vl+vl_temp;

        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven

            vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
            AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5_Veven_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            #No sign for U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,10]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*vl_temp[-1,1,3,5,7,9];
            vl=vl+vl_temp;

        end
    end


    return vl;
end
v_init=TensorMap(randn, space(AA2,1)*space(AA2,1)*space(AA2,1)*space(AA2,1)*space(AA2,1),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1));
v_init=permute(v_init,(1,2,3,4,5,6,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(l_Vodd,l_Veven,x);
#@time _=contraction_fun_R(v_init);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 2,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)


v_init=TensorMap(randn, space(AA2,3)*space(AA2,3)*space(AA2,3)*space(AA2,3)*space(AA2,3),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1)');
v_init=permute(v_init,(6,1,2,3,4,5,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(l_Vodd,l_Veven,x);
#@time _=contraction_fun_L(v_init);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 2,:LM,Arnoldi(krylovdim=10));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4

println(eul)





@tensor VL[:]:=VL[-1,1,2,3,4,5]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9]*U_L[5,-10,-11];#dummy, R1',R1,R2',R2,R3',R3,R4',R4
VL=permute(VL,(1,2,4,6,8,10,3,5,7,9,11,));#dummy, R1',R2',R3',R4',R5', R1,R2,R3,R4,R5


@tensor VR[:]:=VR[1,2,3,4,5,-11]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4]*U_L'[-9,-10,5];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(2,4,6,8,10,1,3,5,7,9,11,));#L1,L2,L3,L4,L5,L1',L2',L3',L4',L5',dummy




@tensor H[:]:=VL[1,-1,-2,-3,-4,-5,2,3,4,5,6]*VR[2,3,4,5,6,-6,-7,-8,-9,-10,1];#R1',R2',R3',R4',R5' ,L1',L2',L3',L4',L5'



eu,ev=eig(H,(1,2,3,4,5,),(6,7,8,9,10,));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,5,6,));#L1',L2',L3',L4',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,6);#L2',L1',L3',L4',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,6);#L2',L3',L1',L4',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,6);#L2',L3',L4',L1',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,6);#L2',L3',L4',L5',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,5,-1]*ev[1,2,3,4,5,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
k_phase=diag(k_phase);
eu_set1=eu[order];
k_phase_set1=k_phase[order];
Qn_set1=Qn[order];
Spin_set1=Spin[order]


####################################
VR=evr[2];#L1,L2,L3,L4,dummy
VL=evl[2];#dummy,R1,R2,R3,R4

@tensor VL[:]:=VL[-1,1,2,3,4,5]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9]*U_L[5,-10,-11];#dummy, R1',R1,R2',R2,R3',R3,R4',R4
VL=permute(VL,(1,2,4,6,8,10,3,5,7,9,11,));#dummy, R1',R2',R3',R4',R5', R1,R2,R3,R4,R5


@tensor VR[:]:=VR[1,2,3,4,5,-11]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4]*U_L'[-9,-10,5];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(2,4,6,8,10,1,3,5,7,9,11,));#L1,L2,L3,L4,L5,L1',L2',L3',L4',L5',dummy




@tensor H[:]:=VL[1,-1,-2,-3,-4,-5,2,3,4,5,6]*VR[2,3,4,5,6,-6,-7,-8,-9,-10,1];#R1',R2',R3',R4',R5' ,L1',L2',L3',L4',L5'



eu,ev=eig(H,(1,2,3,4,5,),(6,7,8,9,10,));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,5,6,));#L1',L2',L3',L4',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,6);#L2',L1',L3',L4',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,6);#L2',L3',L1',L4',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,6);#L2',L3',L4',L1',L5',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,6);#L2',L3',L4',L5',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,5,-1]*ev[1,2,3,4,5,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
k_phase=diag(k_phase);
eu_set2=eu[order];
k_phase_set2=k_phase[order];
Qn_set2=Qn[order];
Spin_set2=Spin[order]


##########################
eu=vcat(eu_set1,eu_set2);
k_phase=vcat(k_phase_set1,k_phase_set2);
Qn=vcat(Qn_set1,Qn_set2);
Spin=vcat(Spin_set1,Spin_set2);

matwrite("ES_Gutzwiller_M1_Nv5"*".mat", Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Qn"=>Qn,
    "Spin"=>Spin
); compress = false)



