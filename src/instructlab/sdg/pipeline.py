# SPDX-License-Identifier: Apache-2.0
# Third Party
from concurrent.futures import ThreadPoolExecutor, Executor, Future
from queue import Queue
from typing import List

import datasets

from datasets import Dataset

# Local
from .iterblock import IterBlock
from .logger_config import setup_logger

logger = setup_logger(__name__)


class Pipeline:
    def __init__(self, chained_blocks: list) -> None:
        """
        Initialize the Pipeline class with a configuration dictionary.
        config_dict: the run config py or yaml loaded into a dictionary
        """
        # pipeline config is the run configuration that consists of the pipeline steps
        self.chained_blocks = chained_blocks

    def generate(self, dataset: Dataset, *, max_workers: int = None) -> Dataset:
        thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        results: List[Dataset] = []
        future_queue = Queue()

        # future_queue.put(
        #     thread_pool.submit(
        #         self._generate_step, dataset, thread_pool, results, future_queue, 0
        #     )
        # )

        self._generate_step(dataset, thread_pool, results, future_queue, 0)

        # Wait for the future queue to be empty: then work should have stopped
        while future_queue.qsize() > 0:
            # Getting the next item should never block
            future = future_queue.get(block=False)
            # ... But waiting on it will
            _ = future.result()

        thread_pool.shutdown(wait=True)
        return datasets.concatenate_datasets(results)

    @staticmethod
    def _drop_duplicates(dataset, cols):
        """
        Drop duplicates from the dataset based on the columns provided.
        """
        df = dataset.to_pandas()
        df.drop_duplicates(subset=cols, inplace=True)
        return Dataset.from_pandas(df)

    def _generate_step(
        self, dataset: Dataset, thread_pool: Executor, results: List[Dataset], future_queue: Queue[Future], index: int = 0
    ) -> None:
        """
        Generate the dataset by running the pipeline steps.
        dataset: the input dataset
        """
        block_prop = self.chained_blocks[index]
        block_type = block_prop["block_type"]
        block_config = block_prop["block_config"]
        drop_columns = block_prop.get("drop_columns", None)
        gen_kwargs = block_prop.get("gen_kwargs", {})
        drop_duplicates_cols = block_prop.get("drop_duplicates", False)

        if block_type == IterBlock:
            block_kwargs = block_config.pop("block_kwargs")
            block = block_type(**block_config, block_kwargs=block_kwargs)
        else:
            block = block_type(**block_config)

        logger.info("Running block: %s", block_config["block_name"])
        logger.info(dataset)

        next_datasets = block.generate(dataset, **gen_kwargs)

        # Allow blocks to return Union[Dataset, Iterator[Dataset]]
        single_output = isinstance(next_datasets, Dataset)
        if single_output:
            next_datasets = [next_datasets]

        next_index = index + 1
        finished = next_index == len(self.chained_blocks)
        for dataset in next_datasets:
            if drop_columns:
                drop_columns = [col for col in drop_columns if col in dataset.column_names]
                dataset = dataset.remove_columns(drop_columns)
            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

            if finished:
                results.append(dataset)
            # elif single_output:
            #     # Run in same thread - TBD
            #     self._generate_step(dataset, thread_pool, results, index=next_index)
            else:
                # Run in new thread
                future_queue.put_nowait(
                    thread_pool.submit(
                        self._generate_step, dataset, thread_pool, results, future_queue, index=next_index
                    )
                )
