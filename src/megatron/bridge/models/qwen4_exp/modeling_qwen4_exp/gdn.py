import torch
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    """Keep SiLU convolution activation independent of the GDN output gate."""

    def _apply_gated_norm(self, hidden_states, gate):
        dtype = hidden_states.dtype
        normalized = self.out_norm(hidden_states.reshape(-1, hidden_states.shape[-1]))
        gate = gate.reshape(-1, gate.shape[-1]).float()
        if self.config.gdn_output_gate == "sigmoid":
            gate = torch.sigmoid(gate)
        elif self.config.gdn_output_gate in ("silu", "swish"):
            gate = torch.nn.functional.silu(gate)
        else:
            raise ValueError(f"Unsupported GDN output gate: {self.config.gdn_output_gate}")
        return (normalized * gate).to(dtype)
