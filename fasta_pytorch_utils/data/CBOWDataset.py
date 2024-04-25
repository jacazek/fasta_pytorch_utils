import math
import torch
import numpy
from fasta_utils import FastaFileReader


class CBOWDataset(torch.utils.data.IterableDataset):
    def __init__(self, fasta_file, tokenizer, vocabulary, device="cpu", dtype=torch.float32, window_size=7,
                 index_file=None):
        super(CBOWDataset).__init__()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.fasta_file = fasta_file
        self.index_file = index_file if index_file is not None else fasta_file + ".fai"
        self.device = device
        self.dtype = dtype
        self.window_size = window_size
        self.window_middle = math.floor(window_size / 2)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with FastaFileReader(self.fasta_file, index_file=self.index_file) as fasta_file_reader:
            data_reader = None
            if worker_info is None:
                data_reader = fasta_file_reader.read_all()
            else:
                if self.index_file is None:
                    raise Exception("index file must be specified when using multiple workers")
                number_of_sequences = fasta_file_reader.get_index_table_length()
                index = worker_info.id % number_of_sequences
                data_reader = fasta_file_reader.read_at_index(index)

            for header, sequence in data_reader:
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