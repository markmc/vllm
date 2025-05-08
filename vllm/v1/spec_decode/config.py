# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from vllm.transformers_utils.config import get_hf_file_to_dict

ALGORITHMS = [
    "eagle",
    "eagle3",
]


class UnknownSpeculativeAlgorithmError(ValueError):

    def __init__(self, algorithm: str):
        msg = (f"Unknown speculative decoding algorithm '{algorithm}'. "
               f"Known algorithms are: ', '.join(ALGORITHMS)")
        super().__init__(msg)


class SpeculatorConfig:

    def __init__(self, algorithm: str):
        self.algorithm = algorithm

    def validate(self):
        if self.algorithm not in ALGORITHMS:
            raise UnknownSpeculativeAlgorithmError(self.algorithm)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SpeculatorConfig":
        spec_config = cls(algorithm=config["speculative_algorithm"])
        spec_config.validate()
        return spec_config


def has_speculator_config(model: str, revision: Optional[str] = 'main'):
    config_dict = get_hf_file_to_dict("config.json", model, revision)
    return "speculator_config" in config_dict
