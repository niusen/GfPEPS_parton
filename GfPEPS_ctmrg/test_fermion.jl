using JLD2
using TensorKit
cd(@__DIR__)

A=load("A.jld2")["A"]

println(A)

println(permute(A,(2,3,),(1,)))
println(A')

@tensor Norm[:]:=A'[2,3,1]*A[1,2,3]