from __future__ import annotations
import torch
from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
import numpy as np
import zonopy.internal as zpi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope as PZType
    from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType

def remove_dependence_and_compress(
        Set: Union[PZType, BPZType, MPZType, BMPZType],
        id: np.ndarray
        ) -> Union[PZType, BPZType, MPZType, BMPZType]:
    
    id_idx = np.any(np.expand_dims(Set.id,1) == id, axis=1)

    has_val = torch.any(Set.expMat[:,id_idx] != 0, dim=1)
    dn_has_val = torch.all(Set.expMat[:,~id_idx] == 0, dim=1)
    ful_slc_idx = torch.logical_and(has_val, dn_has_val)
    
    if isinstance(Set,(polyZonotope,batchPolyZonotope)):
        if zpi.__debug_extra__: assert torch.count_nonzero(ful_slc_idx) <= np.count_nonzero(id_idx)
        c = Set.c
        G = Set.G[...,ful_slc_idx,:]
        ExpMat = Set.expMat[ful_slc_idx][:,id_idx]
        # Instead of reducing Grest now, just leave it
        Z = torch.concat([c.unsqueeze(-2), G, Set.G[...,~ful_slc_idx,:], Set.Grest], dim=-2)
        return type(Set)(Z, G.shape[-2], ExpMat, Set.id[id_idx])
    elif isinstance(Set,(matPolyZonotope,batchMatPolyZonotope)):
        # TODO WHY NO ASSERT HERE
        C = Set.C
        G = Set.G[...,ful_slc_idx,:,:]
        ExpMat = Set.expMat[ful_slc_idx][:,id_idx]
        # Instead of reducing Grest now, just leave it
        Z = torch.concat([C.unsqueeze(-3), G, Set.G[...,~ful_slc_idx,:,:], Set.Grest], dim=-3)
        return type(Set)(Z, G.shape[-3], ExpMat, Set.id[id_idx])
    else:
        raise NotImplementedError

