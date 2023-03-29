using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M1")

include("parton_CTMRG.jl")

include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")


swap_gate_double_layer=false;


M=1;#number of virtual mode
distance=40;
chi=40
tol=1e-6
Guztwiller=true;#add projector



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
CTM, AA, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,swap_gate_double_layer);


Cset=CTM["Cset"];
Tset=CTM["Tset"];

#3x3 term
@tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
@tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
@tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA[2,5,-2,3]*Tset[3][-3,5,4];
@tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
ov_3x3=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_3x3: "*string(ov_3x3));flush(stdout);

#2x3 term
@tensor Norm[:]:=Cset[1][2,1]*Cset[2][5,7]*Cset[3][7,6]*Cset[4][3,2]*Tset[1][1,4,5]*Tset[3][6,4,3];
ov_2x3=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_2x3: "*string(ov_2x3));flush(stdout);

#3x2 term
@tensor Norm[:]:=Cset[1][1,2]*Cset[2][2,3]*Cset[3][6,7]*Cset[4][7,5]*Tset[2][3,4,6]*Tset[4][5,4,1];
ov_3x2=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_3x2: "*string(ov_3x2));flush(stdout);

#2x2 term
@tensor Norm[:]:=Cset[1][1,2]*Cset[2][2,3]*Cset[3][3,4]*Cset[4][4,1];
ov_2x2=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_2x2: "*string(ov_2x2));flush(stdout);

########################################
CTM_ite_nums=50;
@time CTM, AA, U_L,U_D,U_R,U_U=CTMRG(AA,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);


#3x3 term
@tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
@tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
@tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA[2,5,-2,3]*Tset[3][-3,5,4];
@tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
ov_3x3=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_3x3: "*string(ov_3x3));flush(stdout);

#2x3 term
@tensor Norm[:]:=Cset[1][2,1]*Cset[2][5,7]*Cset[3][7,6]*Cset[4][3,2]*Tset[1][1,4,5]*Tset[3][6,4,3];
ov_2x3=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_2x3: "*string(ov_2x3));flush(stdout);

#3x2 term
@tensor Norm[:]:=Cset[1][1,2]*Cset[2][2,3]*Cset[3][6,7]*Cset[4][7,5]*Tset[2][3,4,6]*Tset[4][5,4,1];
ov_3x2=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_3x2: "*string(ov_3x2));flush(stdout);

#2x2 term
@tensor Norm[:]:=Cset[1][1,2]*Cset[2][2,3]*Cset[3][3,4]*Cset[4][4,1];
ov_2x2=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
println("ov_2x2: "*string(ov_2x2));flush(stdout);

###################
using KrylovKit
function correl_TransOp(vl,Tup,Tdown,AAfused)
    if AAfused==[]
        
        @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        
    else
        
        @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        
    end
    return vl
end

function solve_correl_length(n_values,AA_fused,CTM,direction)
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];
    if AA_fused==[]
        println(fuse(space(T1,1)', space(T3,3)))
    else
        println(fuse(space(T1,1)'⊗space(AA_fused,1)', space(T3,3)))
    end
    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,AA_fused)
        if AA_fused==[]
            

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=eu;
            Qspin=eu*0;
            QN=eu*0;

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,0)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+0);
                QN=vcat(QN,0*eu.+2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+0);
                QN=vcat(QN,0*eu.-2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,1)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1);
                QN=vcat(QN,0*eu.+0);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,1)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1);
                QN=vcat(QN,0*eu.+2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,1)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1);
                QN=vcat(QN,0*eu.-2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.+1);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.-1);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((3,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.+3);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)', space(T3,3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.-3);
            end

            eus_abs=abs.(eus);
            @assert maximum(eus_abs)==eus_abs[1]

            eus_abs_sorted=sort(eus_abs,rev=true);
            eus_abs_sorted=eus_abs_sorted/eus_abs_sorted[1];
            Qspin=Qspin[sortperm(eus_abs,rev=true)];
            QN=QN[sortperm(eus_abs,rev=true)];
        else

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=eu;
            Qspin=eu*0;
            QN=eu*0;

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,0)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+0);
                QN=vcat(QN,0*eu.+2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+0);
                QN=vcat(QN,0*eu.-2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,1)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1);
                QN=vcat(QN,0*eu.+0);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,1)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1);
                QN=vcat(QN,0*eu.+2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,1)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1);
                QN=vcat(QN,0*eu.-2);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.+1);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.-1);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((3,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.+3);
            end

            Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3,1/2)=>1);
            vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
            if norm(vl_init)>0
                eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eus=vcat(eus,eu);
                Qspin=vcat(Qspin,0*eu.+1/2);
                QN=vcat(QN,0*eu.-3);
            end

            eus_abs=abs.(eus);
            @assert maximum(eus_abs)==eus_abs[1]

            eus_abs_sorted=sort(eus_abs,rev=true);
            eus_abs_sorted=eus_abs_sorted/eus_abs_sorted[1];
            Qspin=Qspin[sortperm(eus_abs,rev=true)];
            QN=QN[sortperm(eus_abs,rev=true)];
        end

        
        return eus_abs_sorted, Qspin, QN
    end
  
end



function transfer_spectrum(AA_fused,CTM,direction)
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];
    println(fuse(space(T1,1)'⊗space(AA_fused,1)', space(T3,3)))
    if direction=="x"
        @tensor Op[:]:=AA_fused[-2,2,-5,1]*T1[-1,1,-4]*T3[-6,2,-3];
        Op=permute(Op,(1,2,3,),(4,5,6,));
        eu,ev=eigen(Op)
       
        
        return eu,ev
    end
  
end

n_values=10;
eus_abs_sorted, Qspin, QN=solve_correl_length(n_values,AA,CTM,"x");
println(eus_abs_sorted);
println(Qspin)

n_values=10;
eus_abs_sorted_simplified, Qspin_simplified, QN_simplified=solve_correl_length(n_values,[],CTM,"x");
println(eus_abs_sorted_simplified);
println(Qspin_simplified)


eu,ev=transfer_spectrum(AA,CTM,"x");
EU=diag(convert(Array,eu));
EU=sort(abs.(EU));
EU=EU/maximum(EU);
EU=EU[end-100:end];
println(EU);
