using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")

include("GfPEPS_CTMRG.jl")
#include("GfPEPS_model.jl")



chi=10
tol=1e-6




CTM_ite_nums=500;
CTM_trun_tol=1e-10;


U_phy=load("Tensor_M1.jld2")["U_phy"]
T=load("Tensor_M1.jld2")["T"]





U=unitary(fuse(space(T,5)⊗space(T,6)),space(T,5)⊗space(T,6));
@tensor A_fused[:]:=T[-1,-2,-3,-4,1,2]*U[-5,1,2];




init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true);
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);

# @time E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
# @time E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");

# display((E_up+E_down)/3)

display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))




