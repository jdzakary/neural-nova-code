import torch

from model import Actor


def export():
    actor = Actor()
    state_dict = torch.load('results/state/exp3/batch_5899_actor.pt')
    actor.load_state_dict(state_dict)
    dummy_input = torch.ones(3, 9, 9, dtype=torch.float32)
    torch.onnx.export(
        model=actor,
        args=(dummy_input,),
        f='temp.onnx',
        input_names=['observations'],
        output_names=['logits'],
    )
