# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field

from peft.tuners import LoraConfig
from peft.utils import PeftType


@dataclass
class MoELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MoELora`]
    """
    expert_num: int = field(default=4, metadata={"help": "Number of experts in the MoE layer."})

    def __post_init__(self):
        self.peft_type = PeftType.MOELORA
