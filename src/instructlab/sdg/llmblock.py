# SPDX-License-Identifier: Apache-2.0
# Standard
import re
import sys
from typing import Generator, Iterator

# Third Party
from datasets import Dataset

from openai import Completion

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


class LLMBlock(Block):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        block_name,
        config_path,
        client,
        model_id,
        output_cols,
        model_prompt="{prompt}",
        **batch_kwargs,
    ) -> None:
        super().__init__(block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = (
            """{system}\n{introduction}\n{principles}\n{examples}\n{generation}"""
        )
        self.prompt_template = self.prompt_struct.format(**self.block_config)
        self.client = client
        self.model = model_id
        self.model_prompt = model_prompt
        self.output_cols = output_cols
        self.batch_params = batch_kwargs.get("batch_kwargs", {})
        self.defaults = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 12000,
        }

    def _parse(self, generated_string) -> dict:
        matches = {}
        for start_tag, end_tag, output_col in zip(
            self.block_config["start_tags"],
            self.block_config["end_tags"],
            self.output_cols,
        ):
            if not start_tag and not end_tag:
                matches[output_col] = (
                    generated_string.strip() if generated_string else None
                )
            else:
                pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
                all_matches = re.findall(pattern, generated_string, re.DOTALL)
                matches[output_col] = (
                    [match.strip() for match in all_matches] if all_matches else None
                )

        return matches

    def _generate(self, samples, **gen_kwargs) -> Generator[Completion, None, None]:
        prompts = [
            self.model_prompt.format(
                prompt=self.prompt_template.format(**sample).strip()
            )
            for sample in samples
        ]
        return self.client.completions.create(
            prompt=prompts, stream=True, **{**self.defaults, **gen_kwargs}
        )


#        return [choice.text.strip() for choice in response.choices]


    def generate(self, samples: Dataset, **gen_kwargs) -> Iterator[Dataset]:
        """
        Generate the output from the block. This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.

        :return: The parsed output after generation.
        """
        num_inputs = len(samples)

        if not num_inputs:
            logger.warn("ENCOUNTERED EMPTY DATASET!!")
            return []

        num_samples = self.batch_params.get("num_samples", None)

        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * num_inputs)

        # validate the each sample
        for sample in samples:
            if not self._validate(self.prompt_template, sample):
                logger.warn("SAMPLE FAILED VALIDATION")
                return []

        # generate the output
        stream = self._generate(samples, **gen_kwargs)

        col_specs = list(zip(
            self.block_config["start_tags"],
            self.block_config["end_tags"],
            self.output_cols,
        ))

        output_text = [""] * num_inputs
        sent_up_to = [0] * num_inputs
        if num_samples is None:
            num_samples = sys.maxsize
        remaining = [num_samples] * num_inputs

        # [ ([],[]), ([],[]) ]
        outputs = [tuple([] for _ in col_specs) for _ in range(num_inputs)]

        try:
            for chunk in stream:
                choice = chunk.choices[0]
                sample_index = choice.index
                if not remaining[sample_index]:
                    continue

                text = output_text[sample_index] + choice.text
                output_text[sample_index] = text

                start_index = sent_up_to[sample_index]

                for i, (start_tag, end_tag, col_name) in enumerate(col_specs):

                    end_tag_idx = text.rfind(end_tag, start_index)
                    if end_tag_idx != -1:
                        start_tag_idx = text.rfind(start_tag, start_index, end_tag_idx)
                        if start_tag_idx == -1:
                            logger.warn(f"Encountered end tag {end_tag} without corresponding start tag")
                            pass
                        else:
                            output = text[start_tag_idx + len(start_tag): end_tag_idx].strip()
                            sent_up_to[sample_index] = end_tag_idx + len(end_tag)

                            buckets = outputs[sample_index]
                            buckets[i].append(output)

                            if all(buckets):
                                out_texts = [bucket.pop(0) for bucket in buckets]
                                #TODO simplify
                                print(f"Sending dataset for index {sample_index}: {list(out_texts)}")
                                dataset = Dataset.from_list([{
                                    **samples[sample_index], **dict(zip(self.output_cols, out_texts))
                                }])

                                yield dataset

                                remaining[sample_index] -= 1
                                if not any(remaining):
                                    break

        finally:
            # Ensure we abort to free up model server resources
            stream.close()


            #TODO finish reason


        # new_data = []
        # for sample, output in zip(samples, outputs):
        #     parsed_outputs: Dict[str, List[str]] = self._parse(output)
        #     # pylint: disable=consider-using-generator
        #     max_length = max([len(value) for value in parsed_outputs.values()])
        #     for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
        #         new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
        #
        # Dataset.from
        # return Dataset.from_list(new_data)
