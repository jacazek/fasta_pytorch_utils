import os
import torch
import numpy
from fasta_utils import FastaFileReader

class FastaDataset(torch.utils.data.IterableDataset):
    def __init__(self, fasta_file, tokenizer, vocabulary, device="cpu", dtype=torch.float32, index_file=None):
        super(FastaDataset).__init__()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.fasta_file = fasta_file
        self.index_file = index_file if index_file is not None else fasta_file + ".fai"
        self.device = device
        self.dtype = dtype

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with FastaFileReader(self.fasta_file, index_file=self.index_file) as fasta_file_reader:
            data_reader = None
            if worker_info is None or worker_info.num_workers <= 1:
                data_reader = fasta_file_reader.read_all()
            else:
                if self.index_file is None or not os.path.exists(self.index_file):
                    raise Exception("index file must be specified when using multiple workers")
                number_of_sequences = fasta_file_reader.get_index_table_length()
                indices = (np.arange(
                    number_of_sequences / worker_info.num_workers) * worker_info.num_workers + worker_info.id).astype(
                    int)
                indices = indices[indices < number_of_sequences]

                data_reader = fasta_file_reader.read_indices(indices)

            for header, sequence in data_reader:
                sequence = torch.tensor([self.vocabulary[token] for token in
                                         self.tokenizer.tokenize(sequence)],
                                        dtype=torch.long, device=self.device)

                yield sequence, torch.tensor(1 if "gene" in header else 0,
                                             dtype=self.dtype, device=self.device)