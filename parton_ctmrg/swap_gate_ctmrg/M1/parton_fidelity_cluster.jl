using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M1")


include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\fidelity_funs.jl")
include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")


M=2;#number of virtual mode

tol=1e-6
Gutzwiller=true;#add projector



CTM_ite_nums=500;
CTM_trun_tol=1e-10;


theta1=0;
theta2=0.025;


chi=20
cal_fidelity(theta1,theta2,Gutzwiller,M,chi,tol,CTM_ite_nums,CTM_trun_tol)









