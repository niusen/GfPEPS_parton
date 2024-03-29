using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("gauge_flux.jl")
include("double_layer_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")






data=load("swap_gate_Tensor_M2.jld2")
A=data["A"];   #P1,P2,L,R,D,U

A_new=zeros(1,2,2,2,2,2,2,2,2,2,2)*im;
A_new[1,:,:,:,:,:,:,:,:,:,:]=A;
Vdummy=ℂ[U1Irrep](-5=>1);
V=ℂ[U1Irrep](0=>1,1=>1);
# Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((-3,1)=>1);
# V=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((0,0)=>1,(1,1)=>1);
#A_new = TensorMap(A_new, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V'⊗ V'⊗ V'⊗ V');
A_new = TensorMap(A_new, Vdummy' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ← V⊗ V⊗ V⊗ V);


@assert norm(convert(Array,A_new)[1,:,:,:,:,:,:,:,:,:,:]-A)/norm(A)<1e-14
A=A_new; # dummy,P1,P2,L,R,D,U


U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)), space(A,1)⊗space(A,2)⊗space(A,3));
@tensor A[:]:=A[1,2,3,-2,-3,-4,-5,-6,-7,-8,-9]*U_phy1[-1,1,2,3]; # P,L,R,D,U


#Add bond:both parity gate and bond operator
bond=zeros(1,2,2); bond[1,1,2]=1;bond[1,2,1]=1; bond=TensorMap(bond, ℂ[U1Irrep](1=>1)' ← V' ⊗ V');
gate=parity_gate(A,4); @tensor A[:]:=A[-1,-2,-3,1,-5,-6,-7,-8,-9]*gate[-4,1];
gate=parity_gate(A,6); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,-7,-8,-9]*gate[-6,1];
@tensor total_bond[:]:=bond[-1,-5,-9]*bond[-2,-6,-10]*bond[-3,-7,-11]*bond[-4,-8,-12];
@tensor A[:]:=A[-1,-2,-3,1,2,3,4,-8,-9]*total_bond[-10,-11,-12,-13,-4,-5,-6,-7,1,2,3,4];
U_phy2=unitary(fuse(space(A,1)⊗space(A,10)⊗space(A,11)⊗space(A,12)⊗space(A,13)), space(A,1)⊗space(A,10)⊗space(A,11)⊗space(A,12)⊗space(A,13));
@tensor A[:]:=A[1,-2,-3,-4,-5,-6,-7,-8,-9,2,3,4,5]*U_phy2[-1,1,2,3,4,5];
#P,L,R,D,U


###################
#|><R1R2|=|><|R2R1
gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2];  
gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2];  
###################


#group virtual legs on the same legs
U1=unitary(fuse(space(A,2)⊗space(A,3)),space(A,2)⊗space(A,3)); 
U2=unitary(fuse(space(A,8)⊗space(A,9)),space(A,8)⊗space(A,9));

@tensor A[:]:=A[-1,1,2,-3,-4,-5,-6,-7,-8]*U1[-2,1,2];
@tensor A[:]:=A[-1,-2,1,2,-4,-5,-6,-7]*U2'[1,2,-3];
@tensor A[:]:=A[-1,-2,-3,1,2,-5,-6]*U1'[1,2,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1,2]*U2[-5,1,2];


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
boundary_phase_y=0.5;

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




V_odd,V_even=projector_virtual(space(A1,4))

Phy_odd,Phy_even=projector_physical(space(A1,5))

l_Vodd=length(V_odd);
l_Veven=length(V_even);


AA1_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA1_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA1_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA1_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer

AA3_Phyodd_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA3_Phyodd_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA3_Phyodd_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA3_Phyodd_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer

AA3_Phyeven_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA3_Phyeven_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA3_Phyeven_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA3_Phyeven_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer


for cc1=1:l_Vodd
    for cc2=1:l_Vodd
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Vodd_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_odd'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_odd[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyodd_Vodd_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_even'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_even[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyeven_Vodd_Vodd[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Vodd
    for cc2=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Vodd_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_odd'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_odd[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyodd_Vodd_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_even'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_even[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyeven_Vodd_Veven[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Veven
    for cc2=1:l_Vodd
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Veven_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_odd'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_odd[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyodd_Veven_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_even'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_even[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyeven_Veven_Vodd[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Veven
    for cc2=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Veven_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_odd'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_odd[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyodd_Veven_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A3'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_even'[2,-5];
        @tensor A_temp[:]:=A3[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_even[-5,2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA3_Phyeven_Veven_Veven[cc1,cc2]=AA_temp;

    end
end

@tensor Ap_temp[:]:=A2'[-1,-2,-3,-4,1]*Phy_even'[1,-5];
@tensor A_temp[:]:=A2[-1,-2,-3,-4,1]*Phy_even[-5,1];
AA2_Phyeven, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

@tensor Ap_temp[:]:=A2'[-1,-2,-3,-4,1]*Phy_odd'[1,-5];
@tensor A_temp[:]:=A2[-1,-2,-3,-4,1]*Phy_odd[-5,1];
AA2_Phyodd, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

gate_upper=parity_gate(U_L,2); @tensor gate_upper[:]:=gate_upper[2,1]*U_L[-1,1,3]*U_L'[2,3,-2];
gate_lower=parity_gate(U_L,3); @tensor gate_lower[:]:=gate_lower[2,1]*U_L[-1,3,1]*U_L'[3,2,-2];

global AA1_Vodd_Vodd, AA1_Vodd_Veven, AA1_Veven_Vodd, AA1_Veven_Veven
global AA3_Phyeven_Vodd_Vodd, AA3_Phyeven_Vodd_Veven, AA3_Phyeven_Veven_Vodd, AA3_Phyeven_Veven_Veven
global AA3_Phyodd_Vodd_Vodd, AA3_Phyodd_Vodd_Veven, AA3_Phyodd_Veven_Vodd, AA3_Phyodd_Veven_Veven
global AA2_Phyodd, AA2_Phyeven
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
            for parity_P2=0:1
                for parity_P3=0:1
                    vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                    AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Vodd_Vodd[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Vodd_Vodd[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3') 
                    @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor vr_temp[:]:=AA1_temp[-1,2,1,6]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*vr_temp[1,3,5,-4];
                    vr=vr+vr_temp;
                end
            end
        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                    AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Vodd_Veven[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Vodd_Veven[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3')                    
                    @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor vr_temp[:]:=AA1_temp[-1,2,1,6]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*vr_temp[1,3,5,-4];
                    vr=vr+vr_temp;
                end
            end
        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd
            for parity_P2=0:1
                for parity_P3=0:1
                    vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                    AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Veven_Vodd[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Veven_Vodd[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3')                    
                    @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor vr_temp[:]:=AA1_temp[-1,2,1,6]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*vr_temp[1,3,5,-4];
                    vr=vr+vr_temp;
                end
            end
        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                    AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Veven_Veven[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Veven_Veven[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    #No sign for U1*(L1+L2+L3) + U1'*(L1'+L2'+L3') 
                    @tensor vr_temp[:]:=AA1_temp[-1,2,1,6]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*vr_temp[1,3,5,-4];
                    vr=vr+vr_temp;
                end
            end
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
            for parity_P2=0:1
                for parity_P3=0:1
                    vl_temp=deepcopy(vl0);#L1,L2,L3,dummy
                    AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Vodd_Vodd[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Vodd_Vodd[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3') 
                    @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor vl_temp[:]:=AA1_temp[1,2,-2,6]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*vl_temp[-1,1,3,5];
                    vl=vl+vl_temp;
                end
            end
        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
                    AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Vodd_Veven[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Vodd_Veven[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3')                    
                    @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor vl_temp[:]:=AA1_temp[1,2,-2,6]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*vl_temp[-1,1,3,5];
                    vl=vl+vl_temp;
                end
            end
        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd
            for parity_P2=0:1
                for parity_P3=0:1
                    vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
                    AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Veven_Vodd[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Veven_Vodd[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3) + U1'*(L1'+L2'+L3')                    
                    @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                    @tensor vl_temp[:]:=AA1_temp[1,2,-2,6]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*vl_temp[-1,1,3,5];
                    vl=vl+vl_temp;
                end
            end
        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
                    AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
                    if parity_P2==1
                        AA2_temp=deepcopy(AA2_Phyodd);
                    elseif parity_P2==0
                        AA2_temp=deepcopy(AA2_Phyeven);
                    end
                    if parity_P3==1
                        AA3_temp=deepcopy(AA3_Phyodd_Veven_Veven[cc1,cc2]);
                    elseif parity_P3==0
                        AA3_temp=deepcopy(AA3_Phyeven_Veven_Veven[cc1,cc2]);
                    end
                    gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                    if parity_P2==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                    end
                    if parity_P3==1
                        gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                        gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                    end
                    #No sign for U1*(L1+L2+L3) + U1'*(L1'+L2'+L3') 
                    @tensor vl_temp[:]:=AA1_temp[1,2,-2,6]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*vl_temp[-1,1,3,5];
                    vl=vl+vl_temp;
                end
            end
        end
    end


    return vl;
end
v_init=TensorMap(randn, space(AA2_Phyodd,1)*space(AA2_Phyodd,1)*space(AA2_Phyodd,1),Rep[U₁](0=>1));
v_init=permute(v_init,(1,2,3,4,),());#L1,L2,L3,dummy
contraction_fun_R(x)=M_vr(l_Vodd,l_Veven,x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,dummy
VR_aa=deepcopy(VR);

v_init=TensorMap(randn, space(AA2_Phyodd,3)*space(AA2_Phyodd,3)*space(AA2_Phyodd,3),Rep[U₁](0=>1)');
v_init=permute(v_init,(4,1,2,3,),());#dummy,R1,R2,R3
contraction_fun_L(x)=vl_M(l_Vodd,l_Veven,x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=10));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3






@tensor VL[:]:=VL[-1,1,2,3]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7];#dummy, R1',R1,R2',R2,R3',R3
VL=permute(VL,(1,2,4,6,3,5,7,));#dummy, R1',R2',R3', R1,R2,R3


@tensor VR[:]:=VR[1,2,3,-7]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3];#L1',L1,L2',L2,L3',L3,dummy
VR=permute(VR,(2,4,6,1,3,5,7,));#L1,L2,L3,L1',L2',L3',dummy




@tensor H[:]:=VL[1,-1,-2,-3,2,3,4]*VR[2,3,4,-4,-5,-6,1];#R1',R2',R3' ,L1',L2',L3'



eu,ev=eig(H,(1,2,3,),(4,5,6,))
Qn=Get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);
eu=diag(convert(Array,eu));
eu=eu/sum(eu)

println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,));#L1',L2',L3',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,4);#L2',L1',L3',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,4);#L2',L3',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,-1]*ev[1,2,3,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;

order=sortperm(abs.(eu));
k_phase=diag(k_phase);
eu=eu[order];
k_phase=k_phase[order];
Qn=Qn[order];

if y_anti_pbc
    matwrite("ES_freefermion_M2_Nv3_APBC"*".mat", Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn
    ); compress = false)
else
    matwrite("ES_freefermion_M2_Nv3"*".mat", Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn
    ); compress = false)
end

