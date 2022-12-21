using TensorKit

ℂ[SU2Irrep](0=>3)
SU2Space(0=>1)
FermionNumber ⊠ SU2Irrep

VfSu2=ℂ[FermionNumber ⊠ SU2Irrep]((0,0)=>1, (1,1/2)=>1,(2,0)=>1,(2,1)=>1)



t1 = TensorMap(randn, VfSu2 ⊗ VfSu2 ← VfSu2);
unitary(fuse(VfSu2 ⊗ VfSu2),VfSu2 ⊗ VfSu2)

permute(t1,(2,3,),(1,))


VfSu2=ℂ[FermionNumber ⊠ SU2Irrep]((0,0)=>1, (1,1/2)=>1)

t2 = TensorMap(randn, VfSu2 ← VfSu2);

convert(Dict, t2)
convert(Dict, permute(t2,(2,),(1,)))