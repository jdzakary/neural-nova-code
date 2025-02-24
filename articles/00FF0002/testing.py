import torch

from transformer import SharedActorCritic


def export():
    actor = SharedActorCritic(
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
    )
    state_dict = torch.load('results/state/exp1/batch_237.pt')
    actor.load_state_dict(state_dict)
    dummy_input = torch.ones(3, 9, 9, dtype=torch.float32)
    torch.onnx.export(
        model=actor,
        args=(dummy_input,),
        f='temp.onnx',
        input_names=['observations'],
        output_names=['logits', 'state_value'],
    )


if __name__ == '__main__':
    export()
