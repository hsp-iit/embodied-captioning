import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torcheval.metrics.text import Perplexity


class CaptioningPredictor(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.perplexity = 0.0
        self.outputs = {}
        if cfg is not None:
            self.input_height, self.input_width = cfg.height, cfg.width

    def pre_process_input(self, inputs):
        pass

    def forward(self, inputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def backward(self):
        pass

    def configure_optimizers(self):
        pass

    def return_probabilities(self):
        pass

    def compute_perplexity(self, logits=None):
        if logits is None:
            # logits: (n_samples, seq_len, vocab_size)
            logits_temp = self.outputs["logits"]
            logits = torch.empty((logits_temp[0].shape[0], len(logits_temp), logits_temp[0].shape[1]))
            for ind in range(logits_temp[0].shape[0]):
                for seq in range(len(logits_temp)):
                    logits[ind, seq] = logits_temp[seq][ind]
        probs = F.softmax(logits, dim=-1)
        probs = torch.max(probs, dim=-1).values
        sum_log_probs = -probs.log().sum()
        num_tokens = probs.shape[1]
        self.perplexity = torch.exp(sum_log_probs / num_tokens).double()
        return self.perplexity

    def post_process_output(self, outputs):
        pass

    def print_caption(self):
        print(self.outputs["text"])


if __name__ == '__main__':
    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Load model
    model = CaptioningPredictor()
    model.eval()
    model.to(device)

    ############ Test perplexity #############
    # Examples modified from https://pytorch.org/torcheval/main/generated/torcheval.metrics.Perplexity.html
    input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]], [[0.5659, 0.0025, 0.0104]],
                          [[0.9097, 0.0577, 0.7947]]])
    target = torch.tensor([[1], [1], [0], [0]])
    perplexity = Perplexity()
    perplexity.reset()
    perplexity.update(input, target)
    ppl_library = perplexity.compute()
    ppl = model.compute_perplexity(input.permute(1, 0, 2))
    assert torch.isclose(ppl, ppl_library,
                         rtol=1e-03), "Wrong perplexity computation!"

    input = torch.tensor([[[0.5659, 0.0025, 0.0104]], [[0.9097, 0.0577, 0.7947]]])
    target = torch.tensor([[0], [0]])
    perplexity = Perplexity()
    perplexity.reset()
    perplexity.update(input, target)
    ppl_library = perplexity.compute()
    ppl = model.compute_perplexity(input.permute(1, 0, 2))
    assert torch.isclose(ppl, ppl_library,
                         rtol=1e-03), "Wrong perplexity computation!"

    input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]]])
    target = torch.tensor([[1], [1]])
    perplexity = Perplexity()
    perplexity.reset()
    perplexity.update(input, target)
    ppl_library = perplexity.compute()
    ppl = model.compute_perplexity(input.permute(1, 0, 2))
    assert torch.isclose(ppl, ppl_library,
                         rtol=1e-03), "Wrong perplexity computation!"
    print("All test passed!!")
