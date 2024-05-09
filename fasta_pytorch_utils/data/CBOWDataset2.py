import math
import torch
import torch.utils.data
from fasta_utils import FastaFileReader


class CBOWDataset2(torch.utils.data.IterableDataset):
    def __init__(self, fasta_files, tokenizer, vocabulary, device="cpu", dtype=torch.float32, window_size=7, rank=0,
                 device_count=1):
        super(CBOWDataset2).__init__()
        self.rank = rank
        self.device_count = device_count
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.fasta_files = fasta_files
        self.device = device
        self.dtype = dtype
        self.window_size = window_size
        self.window_middle = math.floor(window_size / 2)

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        for i in range(int(len(self.fasta_files) / self.device_count)):
            fasta_file = self.fasta_files[self.rank * self.device_count + i]
            if fasta_file is not None:
                worker_info = torch.utils.data.get_worker_info()

                with FastaFileReader(fasta_file) as fasta_file_reader:
                    if worker_info is None:
                        data_reader = fasta_file_reader.read_all()
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

                    else:
                        number_of_sequences = fasta_file_reader.get_index_table_length()
                        for sequence_number in range(int(number_of_sequences / worker_info.num_workers)):
                            if sequence_number < number_of_sequences:
                                # index = self.rank * worker_info.num_workers + worker_info.id % number_of_sequences
                                data_reader = fasta_file_reader.read_at_index(sequence_number)

                                for header, sequence in data_reader:
                                    iterator = iter(self.tokenizer.tokenize(sequence))
                                    window = [self.vocabulary[next(iterator)] for _ in range(self.window_size)]
                                    try:
                                        while True:
                                            yield (
                                            torch.tensor(window[:self.window_middle] + window[self.window_middle + 1:],
                                                         dtype=torch.long),
                                            torch.tensor(window[self.window_middle], dtype=torch.long))

                                            # Slide the window by one element
                                            window.pop(0)
                                            window.append(self.vocabulary[next(iterator)])

                                    except StopIteration:
                                        # Handle the end of the streaming data
                                        while len(window) == self.window_size:
                                            yield (
                                            torch.tensor(window[:self.window_middle] + window[self.window_middle + 1:],
                                                         dtype=torch.long), torch.tensor(window[self.window_middle],
                                                                                         dtype=torch.long))  # Return the remaining window
                                            window.pop(0)

        # sequence = self.sequence_queue.get(True)
        # while sequence is not None:
        #     iterator = iter(self.tokenizer.tokenize(sequence))
        #     window = [self.vocabulary[next(iterator)] for _ in range(self.window_size)]
        #     try:
        #         while True:
        #             yield (torch.tensor(window[:self.window_middle] + window[self.window_middle + 1:],
        #                                 dtype=torch.long),
        #                    torch.tensor(window[self.window_middle], dtype=torch.long))
        #
        #             # Slide the window by one element
        #             window.pop(0)
        #             window.append(self.vocabulary[next(iterator)])
        #
        #     except StopIteration:
        #         # Handle the end of the streaming data
        #         while len(window) == self.window_size:
        #             yield (torch.tensor(window[:self.window_middle] + window[self.window_middle + 1:],
        #                                 dtype=torch.long), torch.tensor(window[self.window_middle],
        #                                                                 dtype=torch.long))  # Return the remaining window
        #             window.pop(0)
        #     sequence = self.sequence_queue.get(True)
