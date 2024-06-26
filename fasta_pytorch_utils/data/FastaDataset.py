import os
import torch
from fasta_utils import FastaFileReader

def create_dna_sequence_collate_function(vocabulary):
    def collate_fn(batch):
        """
        Collect then pad the sequences into a tensor
        :param batch: the batch
        :return: padded sequences, the original targets, and corresponding original lengths of each sequence
        """
        # sort the batch by length of sequence ascending
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        # unzip the sequences from the corresponding targets
        [sequences, targets] = zip(*batch)

        # make the targets a 2-dimensional batch of size 1, so we can easily support multiple targets later
        # by easily refactoring the dataset and dataloader
        targets = torch.stack([torch.tensor([target]) for target in targets], dim=0)

        # gather the original lengths of the sequence before padding
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)

        """
        The sequences should have already been loaded for cpu manipulation, so we should pad them before
        moving them to the gpu because it is more efficient to pad on the cpu
        """
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=vocabulary["pad"])
        return sequences, targets, lengths

    return collate_fn
class FastaDataset(torch.utils.data.IterableDataset):
    def __init__(self, fasta_files, tokenizer, vocabulary, device="cpu", dtype=torch.float32):
        super(FastaDataset).__init__()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.fasta_files = [(fasta_file, fasta_file+".fai") for fasta_file in fasta_files]
        self.device = device
        self.dtype = dtype

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for fasta_file, index_file in self.fasta_files:
            with FastaFileReader(fasta_file, index_file=index_file) as fasta_file_reader:
                data_reader = None
                if worker_info is None or worker_info.num_workers <= 1:
                    data_reader = fasta_file_reader.read_all()
                else:
                    if self.index_file is None or not os.path.exists(index_file):
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