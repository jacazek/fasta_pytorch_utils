import math
import torch
import torch.utils.data
from fasta_utils import FastaFileReader

from queue import Queue


class QueueConsumer:
    def __init__(self, queue: Queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        next = self.queue.get(True)
        if next is None:
            raise StopIteration
        else:
            return next


class FastaKmerDataset(torch.utils.data.IterableDataset):
    def __init__(self, sequence_queue, tokenizer, vocabulary, device="cpu", dtype=torch.float32, window_size=7,
                 device_count=1):
        super(FastaKmerDataset).__init__()
        self.rank = 0
        self.device_count = device_count
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.sequence_queue = sequence_queue
        self.device = device
        self.dtype = dtype
        self.window_size = window_size
        self.window_middle = math.floor(window_size / 2)

    def set_rank(self, rank):
        self.rank = rank

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # tuples = [(file, index) for index in [10, 11] for file in self.files]
        # print(f"rank: {self.rank}; worker: {worker_info.id}, tuples: {tuples}")
        for tuple in QueueConsumer(self.sequence_queue):
            # print(f"rank: {self.rank}; worker: {worker_info.id}; processing tuple {tuple}")
            file, index = tuple
            with FastaFileReader(file) as fasta_file_reader:
                header, sequence = fasta_file_reader.read_at_index(index)
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
            # print(f"rank: {self.rank}; worker: {worker_info.id}; tuple: {tuple}")
        # print(f"rank: {self.rank}; worker: {worker_info.id}; DONE!")