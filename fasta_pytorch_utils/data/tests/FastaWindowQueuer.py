import queue
import os
import fasta_utils
from fasta_utils.tokenizers.kmertokenizer import KmerTokenizer
from fasta_utils.vocab import Vocab
from fasta_utils import FastaFileReader
from fasta_pytorch_utils.data.FastaWindowQueuer import FastaWindowQueuer, StreamingWindows, PreprocessWindows, FastFileIndexPair, QueueConsumer
import cProfile
import pstats
from multiprocessing import Queue
import numpy as np
import unittest

import time

script_directory = os.path.dirname(os.path.realpath(__file__))
# root_directory = os.path.abspath(os.path.join(script_directory, "../../../"))

class test(unittest.TestCase):
    def test_example(self):
        kmer_size = 7
        stride = 3
        # test_queue = queue.Queue()
        tokenizer = KmerTokenizer(kmer_size=kmer_size, stride=stride)
        vocabulary = Vocab.load(os.path.join(script_directory, "data/7mer-s3-202405182143.pickle"))
        streaming = StreamingWindows(vocabulary, tokenizer, window_size=7)
        preprocessing = PreprocessWindows(vocabulary, tokenizer, window_size=7)
        # dataset = FastaFileQueueDataset(test_queue, embedding_strategy=streaming)

        sequence_file = os.path.join(script_directory, "data/Zm-B97-REFERENCE-NAM-1.0.fa.gz")
        # test_queue.put((sequence_file, 0))

        # with FastaFileReader(sequence_file) as file_reader:
        #     print(f"Sequence metadata: {file_reader.index_table[0]}")
        #     for header, sequence in file_reader.read_at_index(0):
        #         break
        # subsequence = sequence[:10000000]
        output_queue = Queue()
        queuer = FastaWindowQueuer(output_queue, 1, streaming)
        # dataset = FastaSequenceDataset([("", subsequence)], embedding_strategy=preprocessing)

        # profiler = cProfile.Profile()
        # profiler.enable()
        start = time.perf_counter()
        # data = [item for item in dataset]
        producers = queuer.run([FastFileIndexPair(sequence_file, 0)])
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
        count = 0
        for item in QueueConsumer(output_queue):
            count +=1
            if count % 1000000 == 0:
                print(f"got {count} samples")
        end = time.perf_counter()

        # count = 0
        # for item in QueueConsumer(output_queue):
        #     count +=1
        print(f"Streaming queue; time: {end - start:4f}; items queued: {output_queue.qsize()}")
        for producer in producers:
            producer.join()


if __name__ == "__main__":
    unittest.main()