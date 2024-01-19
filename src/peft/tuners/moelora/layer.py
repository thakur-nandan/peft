import math
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora import LoraLayer
from peft.utils.other import transpose


class MoELoraLayer(LoraLayer):

    def __init__(self, num_experts: int, base_layer: nn.Module, **kwargs):
        
        super().__init__(base_layer=base_layer, **kwargs)
        self.num_experts = num_experts
    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, init_router_weights, use_rslora):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        
        # Actual trainable parameters
        self.lora_A.update(nn.ModuleDict({adapter_name: MoELinearA(self.in_features, r, self.num_experts)}))
        self.lora_B.update(nn.ModuleDict({adapter_name: MoELinearB(r, self.out_features, self.num_experts)}))
        
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        
        if init_router_weights:
            self.reset_router_parameters(adapter_name, init_router_weights)
        
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(self.weight.device, dtype=weight.dtype)
            else:
                self.to(self.weight.device)

    def reset_router_parameters(self, adapter_name, init_router_weights):
        if init_router_weights is False:
            return

        if adapter_name in self.router.keys():
            nn.init.normal_(self.router[adapter_name].ff.weight, std=1 / self.r[adapter_name])
    
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            for i in range(self.num_experts):
                if init_lora_weights is True:
                    # initialize A the same way as the default for nn.Linear and B to zero
                    # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name].loraA[i].mlp.weight, a=math.sqrt(5))
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, std=1 / self.r[adapter_name])
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights}")
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)
            
            if adapter_name in self.lora_embedding_A.keys():
                # initialize a the same way as the default for nn.linear and b to zero
                nn.init.zeros_(self.lora_embedding_A[adapter_name].loraA[i].mlp.weight)
                nn.init.normal_(self.lora_embedding_B[adapter_name].loraB[i].mlp.weight)


class MoELoraLinear(nn.Module, MoELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MoELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: Union[bool, str] = True,
        init_router_weights: bool = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        self.num_experts = kwargs.pop("num_experts", 4)
        self.num_experts_per_token = kwargs.pop("num_experts_per_token", 2)

        super().__init__()
        MoELoraLayer.__init__(self, num_experts=self.num_experts, base_layer=base_layer, **kwargs)
        
        # initialize the Router network
        self.expert_embedding = nn.ModuleDict({})
        self.router = nn.ModuleDict({})
        self.expert_embedding.update(nn.ModuleDict({adapter_name: nn.Embedding(1, self.out_features)}))
        self.router.update(nn.ModuleDict({adapter_name: Router(self.out_features, self.num_experts)}))
        
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self._active_adapter = adapter_name
        self.bias = nn.Parameter(base_layer.bias) if base_layer.bias is not None else None
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, init_router_weights, use_rslora)


    def merge(self, task_id) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if self.r[self._active_adapter] > 0:
            expert_weight = self.router[self._active_adapter](self.expert_embedding[self._active_adapter](task_id))
            for i in range(self.num_experts):
                lora_A_weights = self.lora_A[self._active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self._active_adapter].loraB[i].mlp.weight
                self.weight.data += (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self._active_adapter]
                    * expert_weight[..., i]
                )
            self.merged = True


    def unmerge(self, task_id):
        if self._active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self._active_adapter] > 0:
            expert_weight = self.router[self._active_adapter](self.expert_embedding[self._active_adapter](task_id))
            for i in range(self.num_experts):
                lora_A_weights = self.lora_A[self._active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self._active_adapter].loraB[i].mlp.weight
                self.weight.data -= (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self._active_adapter]
                    * expert_weight[..., i]
                )
            self.merged = False


    def forward(self, x: torch.Tensor, **kwargs):
        task_id = torch.tensor(0).to(x.device)
        previous_dtype = x.dtype

        if self._active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        if self.disable_adapters:   # No adapter
            if self.r[self._active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge(task_id)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        elif self.r[self._active_adapter] > 0 and not self.merged:   # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self._active_adapter].loraA[0].weight.dtype)

            # task id should be a tensor for a valid index for nn.Embedding
            expert_weight = self.router[self._active_adapter](self.expert_embedding[self._active_adapter](task_id))

            # implementing from sparse mixture of experts
            _, selected_experts = torch.topk(expert_weight, self.num_experts_per_token)

            for i in range(self.num_experts):   
                if i in selected_experts:
                    result += ( # lora process
                        self.lora_B[self._active_adapter].loraB[i](
                            self.lora_A[self._active_adapter].loraA[i](self.lora_dropout[self._active_adapter](x)),
                        )
                        * self.scaling[self._active_adapter]
                        * expert_weight[..., i].unsqueeze(-1).unsqueeze(0)
                    )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result
    

class MoELinearA(nn.Module):
    '''MoE based LoRA block'''
    def __init__(self, in_features, out_features, num_experts) -> None:

        super().__init__()

        self.num_experts = num_experts
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.num_experts == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.num_experts
        
        for _ in range(self.num_experts):
            self.loraA.append(Expert(self.in_features, self.r))

    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.num_experts):
            outputs.append(self.loraA[i](x))

        return outputs
    

class MoELinearB(nn.Module):
    '''MoE based LoRA block'''
    def __init__(self, in_features, out_features, num_experts) -> None:

        super().__init__()

        self.num_experts = num_experts
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.num_experts == 0
        self.r = self.in_features // self.num_experts
        
        for _ in range(self.num_experts):
            self.loraB.append(Expert(self.r, self.out_features))
    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.num_experts):
            outputs.append(self.loraB[i](x[i]))

        return outputs


class Expert(nn.Module):
    '''Expert block in LoRA'''
    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    
    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)
        return y


class Router(nn.Module):
    '''Router block in LoRA'''
    def __init__(self, input_dim, num_experts):

        super().__init__()
        self.ff = nn.Linear(input_dim, num_experts, bias=False)
        self.activation = nn.Softmax(dim=-1)
    
    def forward(self, x):
        logits = self.ff(x)
        probs = self.activation(logits)
        return probs
