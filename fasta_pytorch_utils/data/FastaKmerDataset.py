import math
import torch
from typing import Iterable
import torch.utils.data
from fasta_utils import FastaFileReader
from abc import ABC, abstractmethod

from queue import Queue


class FastaSequenceQueueConsumer:
    def __init__(self, queue: Queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self) -> tuple[str, str]:
        next = self.queue.get(True)
        if next is None:
            raise StopIteration
        else:
            return next


class FastaFileQueueConsumer:
    def __init__(self, queue: Queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self) -> tuple[str, int]:
        next = self.queue.get(True)
        if next is None:
            raise StopIteration
        else:
            return next


class FastaFileIndexSequenceProvider:
    def __init__(self, file_indexes: Iterable[tuple[str, int, bool]]):
        self.file_indexes = file_indexes

    def __iter__(self):
        for file, index, reverse_compliment in self.file_indexes:
            with FastaFileReader(file) as fasta_file_reader:
                for item in fasta_file_reader.read_at_index(index, reverse_compliment=reverse_compliment):
                    yield item


class EmbeddingStrategy(ABC):
    @abstractmethod
    def get_windows(self, sequence):
        pass


class PreprocessEmbedding(EmbeddingStrategy):
    def __init__(self, vocabulary, tokenizer, window_size, return_tensors=True):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.window_middle = math.floor(window_size / 2)
        self.return_tensors = return_tensors

    def get_windows(self, sequence):
        items = [self.vocabulary[token] for token in self.tokenizer.tokenize(sequence)]
        yield from self.get_context_target(items)

    def get_context_target(self, embedded):
        start = 0
        end = len(embedded) - 8
        window_half = self.window_middle
        while start < end:
            left_start = start
            left_end = left_start + window_half
            right_start = left_end + 1
            right_end = right_start + window_half

            yield (
                embedded[left_start:left_end] + embedded[right_start:right_end],
                embedded[window_half])
            start += 1


class StreamingEmbedding(EmbeddingStrategy):
    def __init__(self, vocabulary, tokenizer, window_size, return_tensors=True):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.window_middle = math.floor(window_size / 2)
        self.return_tensors = return_tensors

    def get_windows(self, sequence):
        iterator = iter(self.tokenizer.tokenize(sequence))
        window = [self.vocabulary[next(iterator)] for _ in range(self.window_size)]
        yield from self.get_context_target(iterator, window)

    def get_context_target(self, iterator, window):
        # if self.return_tensors:
        try:
            while True:
                yield (window[:self.window_middle] + window[self.window_middle + 1:],
                       window[self.window_middle])

                # Slide the window by one element
                window.pop(0)
                window.append(self.vocabulary[next(iterator)])

        except StopIteration:
            # Handle the end of the streaming data
            while len(window) == self.window_size:
                yield (window[:self.window_middle] + window[self.window_middle + 1:],
                       window[self.window_middle])  # Return the remaining window
                window.pop(0)


class FastaSequenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, sequence_provider, embedding_strategy: EmbeddingStrategy, device="cpu", dtype=torch.float32):
        super(FastaSequenceDataset).__init__()
        self.rank = 0
        self.sequence_provider = sequence_provider
        self.device = device
        self.dtype = dtype
        self.embedding_strategy = embedding_strategy

    def set_rank(self, rank):
        self.rank = rank

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for header, sequence in self.sequence_provider:
            for window in self.embedding_strategy.get_windows(sequence):
                yield window


class FastaFileQueueDataset(torch.utils.data.IterableDataset):
    def __init__(self, fasta_file_queue, embedding_strategy: EmbeddingStrategy, device="cpu", dtype=torch.float32):
        super(FastaFileQueueDataset).__init__()
        self.fasta_sequence_dataset = FastaSequenceDataset(
            FastaFileIndexSequenceProvider(FastaFileQueueConsumer(fasta_file_queue)),
            embedding_strategy, device, dtype)

    def set_rank(self, rank):
        self.fasta_sequence_dataset.set_rank(rank)

    def __iter__(self):
        for value in self.fasta_sequence_dataset:
            yield value
