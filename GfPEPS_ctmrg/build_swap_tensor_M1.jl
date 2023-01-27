using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
using Combinatorics
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")


#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=1;#number of virtual modes per bond
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
#init_state="QWZ_M"*string(M)*".jld";#initialize: nothing



W=load(init_state)["W"];
E0=load(init_state)["E0"];




#convention for TensorKit
A=zeros(2,2,2,2,2,2)*im; #P1,P2,L,R,D,U
#If index takes value 1: n=1;   If index takes value 2: n=0;
v0=[1,1,1,1,1,1];
element_pos=collect(combinations(1:P+4*M,Q));
for cc=1:length(element_pos)
    coe=element_pos[cc];
    v=deepcopy(v0);
    for dd=1:length(coe)
        v[coe[dd]]=2;
    end
    m=W[coe,:];
    A[v...]=det(m);

end

save("swap_gate_Tensor_M1.jld2", "A",A);


#convention for matlab
A=zeros(2,2,2,2,2,2)*im; #P1,P2,L,R,D,U
#If index takes value 1: n=1;   If index takes value 2: n=0;
v0=[2,2,2,2,2,2];
element_pos=collect(combinations(1:P+4*M,Q));
for cc=1:length(element_pos)
    coe=element_pos[cc];
    v=deepcopy(v0);
    for dd=1:length(coe)
        v[coe[dd]]=1;
    end
    m=W[coe,:];
    A[v...]=det(m);

end


matwrite("swap_gate_Tensor_M1.mat", Dict(
    "A" => A
); compress = false)