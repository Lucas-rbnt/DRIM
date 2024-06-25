import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple

__all__ = ["MultimodalModel", "MultimodalEncoder", "DRIMSurv", "DRIMU"]


class MultimodalEncoder(nn.Module):
    def __init__(
        self, encoders: Dict[str, nn.Module], devices: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__()
        modalities = list(encoders.keys())
        self._modality_sanity_check(modalities)
        if not devices:
            devices = {k: next(v.parameters()).device for k, v in encoders.items()}

        self.devices = devices
        for key, model in encoders.items():
            setattr(self, f"{key.lower()}_encoder", model.to(self.devices[key]))

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._forward_encoders(x)

    def _forward_encoders(
        self, x: Dict[str, torch.Tensor], unique: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        # put the input tensors to the corresponding device
        x = {k: v.to(self.devices[k]) for k, v in x.items()}
        # forward pass
        if unique:
            x = {k: getattr(self, f"{k.lower()}_encoder_u")(v) for k, v in x.items()}
        else:
            x = {k: getattr(self, f"{k.lower()}_encoder")(v) for k, v in x.items()}

        # normalize embeddings
        x = {k: nn.functional.normalize(v, p=2, dim=-1) for k, v in x.items()}

        return x

    def _put_on_devices(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]],
        key: str,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        x = {k: v.to(self.devices[key]) for k, v in x.items()}
        if mask is not None:
            mask = {k: v.to(self.devices[key]) for k, v in mask.items()}
        return x, mask

    @staticmethod
    def _modality_sanity_check(
        modalities: List[str],
        available_modalities: List[str] = ["RNA", "MRI", "DNAm", "WSI"],
    ) -> None:
        for modality in modalities:
            assert (
                modality in available_modalities
            ), f"The requested modality: {modality} is not available, please choose modalities among {available_modalities}"


class MultimodalModel(MultimodalEncoder):
    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        fusion: nn.Module,
        devices: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(encoders, devices)
        if "fusion" not in self.devices.keys():
            try:
                self.devices["fusion"] = next(fusion.parameters()).device
            except:
                # if fusion do not have parameters, then it is a function
                self.devices["fusion"] = "cuda:0"
        self.fusion = fusion

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.tensor, Dict[str, torch.Tensor]]]:
        x = super().forward(x)

        x, mask = self._put_on_devices(x, mask, "fusion")
        fused = self.fusion(x, mask)
        # if fusion returns a tuple, then it is a masked attention fusion
        if isinstance(fused, tuple):
            fused, _ = fused
        if return_embedding:
            return fused, x
        else:
            return fused


class DRIMSurv(MultimodalEncoder):
    def __init__(
        self,
        encoders_sh: Dict[str, nn.Module],
        encoders_u: Dict[str, nn.Module],
        fusion_s: nn.Module,
        fusion_u: nn.Module,
        devices: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(encoders_sh, devices)
        for key, model in encoders_u.items():
            setattr(self, f"{key.lower()}_encoder_u", model.to(f"{self.devices[key]}"))

        if "fusion_sh" not in self.devices.keys():
            try:
                self.devices["fusion_s"] = next(fusion_s.parameters()).device
            except:
                # if fusion do not have parameters, then it is a function
                self.devices["fusion_s"] = "cuda:0"

        if "fusion_u" not in self.devices.keys():
            try:
                self.devices["fusion_u"] = next(fusion_u.parameters()).device
            except:
                # if fusion do not have parameters, then it is a function
                self.devices["fusion_u"] = "cuda:0"
        self.fusion_s = fusion_s
        self.fusion_u = fusion_u

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.tensor, Dict[str, torch.Tensor]]]:
        x_s = super()._forward_encoders(x)
        x_u = super()._forward_encoders(x, unique=True)

        fused = self.drim_forward(x_s, x_u, mask)
        if isinstance(fused, tuple):
            fused, _ = fused
        if return_embedding:
            return fused, x_s, x_u
        else:
            return fused

    def drim_forward(
        self,
        x_s: Dict[str, torch.Tensor],
        x_u: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.tensor, Dict[str, torch.Tensor]]]:
        x_s, mask = self._put_on_devices(x_s, mask, "fusion_s")
        fused_s = self.fusion_s(x_s, mask)
        if isinstance(fused_s, tuple):
            fused_s, _ = fused_s
        x_t = {**x_u, "shared": fused_s}
        mask.update({"shared": torch.ones_like(next(iter(mask.values())))})
        # put everything on the same device
        x_t, mask = self._put_on_devices(x_t, mask, "fusion_u")
        fused = self.fusion_u(x_t, mask)
        return fused


class DRIMU(DRIMSurv):
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.tensor, Dict[str, torch.Tensor]]]:
        x_s = super()._forward_encoders(x)
        x_u = super()._forward_encoders(x, unique=True)
        if return_embedding:
            return x_s, x_u
        else:
            fused = self.drim_forward(x_s, x_u, mask)
            if isinstance(fused, tuple):
                fused, _ = fused

            return fused
