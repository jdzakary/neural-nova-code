from typing import Dict, Any, Tuple

import tree
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces.space_utils import batch as batch_func
from ray.rllib.core.columns import Columns


class RandomWithMasking(RLModule):

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch)

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch)

    @override(RLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        raise NotImplementedError('Random RlModule: Should not be Trained!!!')

    @override(RLModule)
    def _forward(self, batch, **kwargs) -> Dict[str, Any]:
        mask, batch = self.__preprocess_batch(batch)
        batch[Columns.ACTION_PROB]

    def compile(self, compile_config: TorchCompileConfig):
        pass

    @staticmethod
    def __preprocess_batch(
        batch: Dict[str, TensorType],
        **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        action_mask = batch[Columns.OBS].pop("action_mask")
        batch[Columns.OBS] = batch[Columns.OBS].pop("observations")
        return action_mask, batch
