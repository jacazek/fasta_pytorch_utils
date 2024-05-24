from typing import Iterable, Generic, TypeVar
from queue import Queue
from fasta_utils import FastaFileReader
from fasta_utils.tokenizers import KmerTokenizer
from fasta_pytorch_utils.data.FastaKmerDataset import EmbeddingStrategy
from multiprocessing import Process, Queue
from fasta_utils.vocab import Vocab
import math
import torch
from abc import ABC, abstractmethod


class WindowingStrategy(ABC):
    @abstractmethod
    def get_windows(self, sequence):
        pass


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

class FastFileIndexPair:
    def __init__(self, file: str, index: int):
        self.file = file
        self.index = index

    def __repr__(self):
        return f"FastFileIndexPair(file={self.file!r}, index={self.index!r})"


class FastaFileQueueConsumer:
    def __init__(self, queue: Queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self) -> FastFileIndexPair:
        next = self.queue.get(True)
        if next is None:
            raise StopIteration
        else:
            return next


def producer(file_queue: Queue, output_queue: Queue, windowing_strategy: WindowingStrategy):
    try:
        for item in FastaFileQueueConsumer(file_queue):
            with FastaFileReader(item.file) as fasta_file_reader:
                print(f"producer windowing sequence {item.file}")
                for header, sequence in fasta_file_reader.read_at_index(item.index):
                    for window in windowing_strategy.get_windows(sequence):
                        output_queue.put(window)

        print("producer complete")
    except:
        print("something went wrong")


class FastaWindowQueuer:
    def __init__(self, output_queue: Queue, producer_count, windowing_strategy: WindowingStrategy):
        self.output_queue = output_queue
        self.windowing_strategy = windowing_strategy
        self.producer_count = producer_count

    def run(self, input_files: Iterable[FastFileIndexPair]):
        # start up the specified number of producers
        producer_processes = []
        file_queue = Queue()
        print("Queuing files")
        # Enqueu all files for processing by producers
        for fasta_file_index_pair in input_files:
            file_queue.put(fasta_file_index_pair)
        print(f"creating producers {self.producer_count}")
        # Enqueue a stop flag for all producers so when all files are processed, they stop
        for index in range(self.producer_count):
            file_queue.put(None)
            producer_processes.append(Process(target=producer, args=(file_queue, self.output_queue, self.windowing_strategy)))

        # consumer_process = Process(target=consumer, args=(token_queue,))
        print("starting producers")
        for producer_process in producer_processes:
            producer_process.start()
        # consumer_process.start()

        # print('joining producers')
        # for producer_process in producer_processes:
        #     try:
        #         producer_process.join()
        #     except:
        #         print("something went wrong joinin")
        # print("producer has finished")

        # for i in range(self.consumer_count):
        #     self.output_queue.put(None)

        print("done")
        return producer_processes



class PreprocessWindows(WindowingStrategy):
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
        end = len(embedded) - (self.window_size+1)
        window_half = self.window_middle
        # if self.return_tensors:
        while start < end:
            left_start = start
            left_end = left_start + window_half
            right_start = left_end + 1
            right_end = right_start + window_half

            yield (
                embedded[left_start:left_end] + embedded[right_start:right_end],
                embedded[window_half])
            start += 1

class StreamingWindows(WindowingStrategy):
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


def create_window_queue_collator(vocabulary):
    def collate_fn(batch):
        """
        Collect then pad the sequences into a tensor
        :param batch: the batch
        :return: padded sequences, the original targets, and corresponding original lengths of each sequence
        """
        # sort the batch by length of sequence ascending

        # unzip the sequences from the corresponding targets
        [contexts, targets] = zip(*batch)
        return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class FastaWindowQueueDataset(torch.utils.data.IterableDataset):
    def __init__(self, window_queue, tokenizer, vocabulary, device="cpu", dtype=torch.float32, index_file=None):
        super(FastaWindowQueueDataset).__init__()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.window_queue = window_queue
        self.index_file = index_file if index_file is not None else fasta_file + ".fai"
        self.device = device
        self.dtype = dtype

    def __iter__(self):
        yield from QueueConsumer(self.window_queue)



