import torch

from model.cnn import Actor


def export():
    actor = Actor()
    state_dict = torch.load('results/state/exp4/batch_183_actor.pt')
    actor.load_state_dict(state_dict)
    dummy_input = torch.ones(3, 9, 9, dtype=torch.float32)
    torch.onnx.export(
        model=actor,
        args=(dummy_input,),
        f='exports/exp4_actor.onnx',
        input_names=['observations'],
        output_names=['logits'],
    )


if __name__ == '__main__':
    export()
