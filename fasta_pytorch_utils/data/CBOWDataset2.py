import math
import torch
import torch.utils.data
from fasta_utils import FastaFileReader


class CBOWDataset2(torch.utils.data.IterableDataset):
    def __init__(self, sequence_queue, tokenizer, vocabulary, device="cpu", dtype=torch.float32, window_size=7, rank=0):
        super(CBOWDataset2).__init__()
        self.rank = rank
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.sequence_queue = sequence_queue
        self.device = device
        self.dtype = dtype
        self.window_size = window_size
        self.window_middle = math.floor(window_size / 2)

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        sequence = self.sequence_queue.get(True)
        while sequence is not None:
            iterator = iter(self.tokenizer.tokenize(sequence))
            window = [self.vocabulary[next(iterator)] for _ in range(self.window_size)]
            try:
                while True:
                    yield (torch.tensor(window[:self.window_middle] + window[self.window_middle + 1:],
                                        dtype=torch.long),
                           torch.tensor(window[self.window_middle], dtype=torch.long))

                    # Slide the window by one element
                    window.pop(0)
                    window.append(self.vocabulary[next(iterator)])

            except StopIteration:
                # Handle the end of the streaming data
                while len(window) == self.window_size:
                    yield (torch.tensor(window[:self.window_middle] + window[self.window_middle + 1:],
                                        dtype=torch.long), torch.tensor(window[self.window_middle],
                                                                        dtype=torch.long))  # Return the remaining window
                    window.pop(0)
            sequence = self.sequence_queue.get(True)