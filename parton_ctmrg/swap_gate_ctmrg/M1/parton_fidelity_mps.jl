using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M1")


include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\fidelity_funs.jl")
include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")

swap_gate_double_layer=true;
M=1;#number of virtual mode

tol=1e-6
Gutzwiller=true;#add projector



CTM_ite_nums=500;
CTM_trun_tol=1e-10;


theta1=0;
theta2=0.025;


chi=40
forced_steps=50;

filenm1="state1.jld2";
filenm2="state2.jld2";

#cal_fidelity(filenm1,filenm2,Gutzwiller,M,chi,tol,CTM_ite_nums,CTM_trun_tol,forced_steps,swap_gate_double_layer)

cal_fidelity_MPS(filenm1,filenm2,Gutzwiller,M,chi,tol,CTM_ite_nums,CTM_trun_tol,forced_steps,swap_gate_double_layer)








