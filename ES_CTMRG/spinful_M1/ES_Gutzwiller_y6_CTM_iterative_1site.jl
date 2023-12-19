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
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg\\swap_gate_ctmrg\\M1\\GfPEPS_CTMRG.jl")

include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\ES_algorithms.jl")


chi=20
tol=1e-6
CTM_ite_nums=500;
CTM_trun_tol=1e-10;



filenm="Optim_1site_D_4_chi_90.jld2";

data=load(filenm);
A=data["A"];



y_anti_pbc=false;
boundary_phase_y=0.5;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,4,2*pi/6*boundary_phase_y);
    @tensor A[:]:=A[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end


#############################


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,"PBC",true);
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);

N=6;
EH_n=30;
decomp=false;
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\ES_algorithms.jl")

ES_CTMRG_ED_SU2(CTM,U_L,U_D,U_R,U_U,1,chi,N,EH_n,decomp,y_anti_pbc);




