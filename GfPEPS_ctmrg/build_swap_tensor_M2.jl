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
M=2;#number of virtual modes per bond
M_initial=2;
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
init_state="Hofstadter_N2_M"*string(M_initial)*".jld";#initialize: nothing
#init_state="QWZ_M"*string(M)*".jld";#initialize: nothing

if M_initial==2

W=load(init_state)["W"];
E0=load(init_state)["E0"];


#####################
elseif M_initial==1

W_init=load(init_state)["W"];

Q_initial=2*M_initial+filling;
W=[W_init[1:P+M_initial,:];zeros(M-M_initial,Q_initial);W_init[P+M_initial+1:P+2*M_initial,:];zeros(M-M_initial,Q_initial);W_init[P+2*M_initial+1:P+3*M_initial,:];zeros(M-M_initial,Q_initial);W_init[P+3*M_initial+1:P+4*M_initial,:];zeros(M-M_initial,Q_initial)];
W=[W zeros(size(W,1),2*(M-M_initial))];
for cc=1:M-M_initial
    W[P+M_initial+cc,Q_initial+cc]=1;
    W[P+2*M+M_initial+cc,Q_initial+M-M_initial+cc]=1;
end
@assert norm(W'*W-I(Q))<1e-12;
#####################
# W[4,4]=1/2;
# W[6,4]=1/2;
# W[8,4]=1/2;
# W[10,4]=1/2;
# W[4,5]=1/2;
# W[6,5]=1/2;
# W[8,5]=-1/2;
# W[10,5]=-1/2;
pp=Matrix(I,10,10);pp[3,3]=0;pp[4,4]=0;pp[3,4]=1;pp[4,3]=1;
W=pp*W;
pp=Matrix(I,10,10);pp[5,5]=0;pp[6,6]=0;pp[5,6]=1;pp[6,5]=1;
W=pp*W;
pp=Matrix(I,10,10);pp[7,7]=0;pp[8,8]=0;pp[7,8]=1;pp[8,7]=1;
W=pp*W;
pp=Matrix(I,10,10);pp[9,9]=0;pp[10,10]=0;pp[9,10]=1;pp[10,9]=1;
W=pp*W;

#####################
end

#convention for TensorKit
A=zeros(2,2,2,2,2,2,2,2,2,2)*im; #P1,P2,L,R,D,U
#If index takes value 1: n=1;   If index takes value 2: n=0;
v0=[1,1,1,1,1,1,1,1,1,1];
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

save("swap_gate_Tensor_M2.jld2", "A",A);


#convention for matlab
A=zeros(2,2,2,2,2,2,2,2,2,2)*im; #P1,P2,L,R,D,U
#If index takes value 1: n=1;   If index takes value 2: n=0;
v0=[2,2,2,2,2,2,2,2,2,2];
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


matwrite("swap_gate_Tensor_M2.mat", Dict(
    "A" => A
); compress = false)