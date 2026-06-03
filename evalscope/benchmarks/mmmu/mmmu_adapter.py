import ast
import base64
import copy
import hashlib
import io
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content
from evalscope.api.messages.content import ContentImage
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.pruning import DEFAULT_SEED
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt

logger = get_logger()

SUBSET_LIST = [
    'Accounting',
    'Agriculture',
    'Architecture_and_Engineering',
    'Art',
    'Art_Theory',
    'Basic_Medical_Science',
    'Biology',
    'Chemistry',
    'Clinical_Medicine',
    'Computer_Science',
    'Design',
    'Diagnostics_and_Laboratory_Medicine',
    'Economics',
    'Electronics',
    'Energy_and_Power',
    'Finance',
    'Geography',
    'History',
    'Literature',
    'Manage',
    'Marketing',
    'Materials',
    'Math',
    'Mechanical_Engineering',
    'Music',
    'Pharmacy',
    'Physics',
    'Psychology',
    'Public_Health',
    'Sociology',
]

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

OPEN_PROMPT = """
Solve the following problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \\boxed command.

"""  # noqa: E501

MULTI_CHOICE_TYPE = 'multiple-choice'
OPEN_TYPE = 'open'


@register_benchmark(
    BenchmarkMeta(
        name='mmmu',
        pretty_name='MMMU',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

MMMU (Massive Multi-discipline Multimodal Understanding) is a comprehensive benchmark designed to evaluate multimodal models on expert-level tasks requiring college-level subject knowledge and deliberate reasoning. It covers 30 subjects across 6 core disciplines.

## Task Description

- **Task Type**: Multimodal Question Answering (Multiple-Choice and Open-Ended)
- **Input**: Questions with diverse images (charts, diagrams, maps, tables, etc.)
- **Output**: Answer letter (MC) or free-form text (Open)
- **Disciplines**: Art & Design, Business, Science, Health & Medicine, Humanities, Tech & Engineering

## Key Features

- 11.5K meticulously collected multimodal questions
- From college exams, quizzes, and textbooks
- 30 subjects and 183 subfields covered
- 30 heterogeneous image types (charts, diagrams, music sheets, chemical structures, etc.)
- Tests both perception and expert-level reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Supports both multiple-choice and open-ended question types
- Multiple images per question supported (up to 7)
- For open questions: "ANSWER: [ANSWER]" format expected
- Evaluates on validation split (test set requires submission)
""",
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
    )
)
class MMMUAdapter(VisionLanguageAdapter):
    """
    example1:{
        'question': '<image 1> illustrate a walk and a cycle. We can easily represent walking as?'
        'options': "['a line', 'a curve', 'a plane', 'a surface']",
        'image_1': {'bytes': b'...'},
        'question_type': 'multiple-choice',
    }
    example2:{
        'question': 'Select the correct tuning of Violin.',
        'options': "[<image 1>, <image 2>, <image 3>, <image 4>]",
        'image_1': {'bytes': b'...'},
        'image_2': {'bytes': b'...'},
        'image_3': {'bytes': b'...'},
        'image_4': {'bytes': b'...'},
        'question_type': 'multiple-choice',
    }
    example3:{
        'question': 'Each of seven students has chosen three courses from ten options, and must sit an exam for each of his or her three choices. Two students sitting the same exam must do so at the same time, but no student can sit more than one exam in the same day. The table of choices is given in <image 1>. Find the smallest number of days required to schedule the exams. Return only the number of days.',
        'options': '[]',
        'image_1': {'bytes': b'...'},
        'question_type': 'open',
    }
    """  # noqa: E501
    MAX_IMAGES: int = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question_type = record['question_type']
        content_list, answers_list = self.create_content_and_answers_list(record)

        metadata = {
            'id': record['id'],
            'question': record['question'],
            'question_type': record['question_type'],
            'subfield': record['subfield'],
            'explanation': record['explanation'],
            'img_type': record['img_type'],
            'topic_difficulty': record['topic_difficulty'],
            'image_count': sum(1 for i in range(self.MAX_IMAGES) if record.get(f'image_{i + 1}')),
        }

        if question_type == MULTI_CHOICE_TYPE:
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                choices=answers_list,
                target=record['answer'],
                metadata=metadata,
            )
        elif question_type == OPEN_TYPE:
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=record['answer'],
                metadata=metadata,
            )
        else:
            raise ValueError(f'Unsupported question type: {question_type}')

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        question_type = task_state.metadata['question_type']
        if question_type == MULTI_CHOICE_TYPE:
            answers = parse_answers(task_state)
            return ''.join(sorted(list(answers)))
        elif question_type == OPEN_TYPE:
            matches = re.findall(r'ANSWER:\s*(.*)', prediction)
            if matches:
                return matches[-1].strip()
            return prediction.strip()
        else:
            raise ValueError(f'Unsupported question type: {question_type}')

    def create_content_and_answers_list(self, record: Dict[str, Any]) -> tuple[List[Content], List[str]]:
        """
        Create a list of content elements and a list of answers from a record.
        Images are inserted at their <image x> placeholder positions in the text.

        Args:
            record (dict): The record containing question, images, and options.

        Returns:
            tuple: A tuple containing:
                - content_list (list): A list of content elements (text and images).
                - answers_list (list): A list of possible answers (for multiple-choice questions).
        """
        question_type = record['question_type']

        # Prepare image map
        image_map: Dict[int, str] = {}
        for i in range(MMMUAdapter.MAX_IMAGES):
            image = record.get(f'image_{i+1}')
            if image:
                image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
                image_map[i + 1] = image_base64

        if question_type == MULTI_CHOICE_TYPE:
            answers_list: List[str] = ast.literal_eval(record['options'])

            # Build prompt text
            full_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)

            # Parse and replace image placeholders
            content_list = self._parse_text_with_images(full_text, image_map)

        else:  # OPEN_TYPE
            answers_list: List[str] = []
            full_text = OPEN_PROMPT.format(question=record['question'])
            content_list = self._parse_text_with_images(full_text, image_map)

        return content_list, answers_list


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_encoder_probe',
        pretty_name='MMMU Encoder Probe',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
MMMU encoder probe for multimodal qualification. It selects a deterministic
image-stress subset from the full MMMU validation dataset and can expand each
selected sample into original plus degraded image variants.
""",
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
        extra_params={
            'probe_ratio': {
                'type': 'float',
                'description': 'Fraction of each MMMU subject subset to keep before perturbation expansion.',
                'value': 0.05,
            },
            'probe_size': {
                'type': 'int | null',
                'description': 'Optional maximum source samples per subject before perturbation expansion.',
                'value': None,
            },
            'perturbations': {
                'type': 'list[str]',
                'description': 'Image variants to run in addition to original.',
                'value': ['downsample', 'jpeg_low'],
            },
            'seed': {
                'type': 'int',
                'description': 'Deterministic tie-break seed.',
                'value': DEFAULT_SEED,
            },
        },
    )
)
class MMMUEncoderProbeAdapter(MMMUAdapter):
    STRESS_KEYWORDS: Dict[str, Tuple[str, float]] = {
        'ocr': ('ocr_text', 5.0),
        'text': ('ocr_text', 3.0),
        'table': ('table_chart', 4.0),
        'chart': ('table_chart', 4.0),
        'plot': ('table_chart', 4.0),
        'graph': ('table_chart', 3.0),
        'diagram': ('diagram_map', 3.5),
        'map': ('diagram_map', 3.0),
        'label': ('dense_label', 3.0),
        'chemical': ('scientific_visual', 3.0),
        'structure': ('scientific_visual', 2.5),
        'medical': ('scientific_visual', 2.5),
        'microscope': ('scientific_visual', 2.5),
        'geometry': ('diagram_map', 2.0),
    }

    def load_dataset(self) -> DatasetDict:
        dataset = super().load_dataset()
        ratio = float(self.extra_params.get('probe_ratio') or 0.05)
        probe_size = self.extra_params.get('probe_size')
        seed = int(self.extra_params.get('seed') or DEFAULT_SEED)
        perturbations = list(self.extra_params.get('perturbations') or [])

        probed = {}
        for subset, subset_dataset in dataset.items():
            samples = list(subset_dataset)
            selected = self._select_samples(samples, ratio=ratio, probe_size=probe_size, seed=seed)
            expanded = []
            for sample in selected:
                expanded.append(self._variant_sample(sample, 'original'))
                for perturbation in perturbations:
                    expanded.append(self._variant_sample(sample, perturbation))
            memory = MemoryDataset(expanded, name=subset_dataset.name, location=subset_dataset.location)
            memory.reindex()
            probed[subset] = memory
            logger.info(f'MMMU encoder probe {subset}: {len(samples)} -> {len(expanded)} variant samples')
        return DatasetDict(probed)

    def _select_samples(self, samples: List[Sample], *, ratio: float, probe_size: Any, seed: int) -> List[Sample]:
        if not samples:
            return []
        keep_n = max(1, int(len(samples) * ratio))
        if probe_size is not None:
            keep_n = min(keep_n, int(probe_size))

        strata: Dict[Tuple[str, ...], List[Tuple[float, str, Sample]]] = defaultdict(list)
        for ordinal, sample in enumerate(samples):
            metadata = sample.metadata or {}
            source_id = str(metadata.get('id') or ordinal)
            strata[self._stratum(sample, seed)].append(
                (-self._stress_score(sample), self._stable_rank(source_id, seed), sample)
            )

        quotas = self._quotas({key: len(value) for key, value in strata.items()}, keep_n)
        selected: List[Sample] = []
        selected_ids = set()
        for stratum in sorted(strata):
            for _, _, sample in sorted(strata[stratum], key=lambda row: (row[0], row[1]))[:quotas[stratum]]:
                selected.append(sample)
                selected_ids.add(str((sample.metadata or {}).get('id') or sample.id))

        if len(selected) < keep_n:
            remaining = []
            for group in strata.values():
                for row in group:
                    sample = row[2]
                    source_id = str((sample.metadata or {}).get('id') or sample.id)
                    if source_id not in selected_ids:
                        remaining.append(row)
            selected.extend(
                sample for _, _, sample in sorted(remaining, key=lambda row: (row[0], row[1]))[:keep_n - len(selected)]
            )
        return selected

    def _stratum(self, sample: Sample, seed: int) -> Tuple[str, ...]:
        metadata = sample.metadata or {}
        return (
            str(metadata.get('question_type') or 'unknown'),
            self._image_type_family(str(metadata.get('img_type') or 'unknown')),
            str(metadata.get('topic_difficulty') or 'unknown'),
            'multi_image' if int(metadata.get('image_count') or 0) > 1 else 'single_image',
            self._text_cluster(sample, seed),
        )

    @staticmethod
    def _quotas(stratum_sizes: Dict[Tuple[str, ...], int], keep_n: int) -> Dict[Tuple[str, ...], int]:
        total = sum(stratum_sizes.values())
        quotas = {key: int((size * keep_n) // max(1, total)) for key, size in stratum_sizes.items()}
        remainders = sorted(((size * keep_n / max(1, total)) - quotas[key], key) for key, size in stratum_sizes.items())
        while sum(quotas.values()) < keep_n:
            changed = False
            for _, key in reversed(remainders):
                if sum(quotas.values()) >= keep_n:
                    break
                if quotas[key] < stratum_sizes[key]:
                    quotas[key] += 1
                    changed = True
            if not changed:
                break
        return quotas

    @staticmethod
    def _image_type_family(value: str) -> str:
        lowered = value.lower()
        if any(token in lowered for token in ('table', 'chart', 'plot', 'graph')):
            return 'table_chart'
        if any(token in lowered for token in ('diagram', 'map', 'geometry')):
            return 'diagram_map'
        if any(token in lowered for token in ('chemical', 'medical', 'microscope', 'structure')):
            return 'scientific_visual'
        if any(token in lowered for token in ('text', 'ocr', 'label')):
            return 'ocr_text'
        return 'other_visual'

    @staticmethod
    def _text_cluster(sample: Sample, seed: int) -> str:
        metadata = sample.metadata or {}
        text = ' '.join(str(metadata.get(key, '')) for key in ('question', 'subfield', 'img_type')).lower()
        tokens = sorted(set(re.findall(r'[a-zA-Z_][a-zA-Z_0-9]+', text)[:160]))
        digest = hashlib.sha256(f'{tokens}:{seed}'.encode('utf-8')).hexdigest()
        return f'cluster_{int(digest[:8], 16) % 12}'

    def _stress_score(self, sample: Sample) -> float:
        tags, score = self._stress_tags_and_score(sample)
        return score

    def _stress_tags_and_score(self, sample: Sample) -> Tuple[List[str], float]:
        metadata = sample.metadata or {}
        text = ' '.join(str(metadata.get(key, ''))
                        for key in ('question', 'img_type', 'subfield', 'explanation')).lower()
        tags = set()
        score = 0.0
        for keyword, (tag, weight) in self.STRESS_KEYWORDS.items():
            if keyword in text:
                tags.add(tag)
                score += weight
        if int(metadata.get('image_count') or 0) > 1:
            tags.add('multi_image')
            score += 2.0
        if metadata.get('question_type') == OPEN_TYPE:
            tags.add('open_answer')
            score += 0.5
        if not tags:
            tags.add('general_visual')
        return sorted(tags), score

    @staticmethod
    def _stable_rank(source_id: str, seed: int) -> str:
        return hashlib.sha256(f'{source_id}:{seed}'.encode('utf-8')).hexdigest()

    def _variant_sample(self, sample: Sample, variant: str) -> Sample:
        variant_sample = copy.deepcopy(sample)
        source_id = str((variant_sample.metadata or {}).get('id') or variant_sample.id)
        variant_sample.metadata = {
            **(variant_sample.metadata or {}),
            'probe_source_id': source_id,
            'probe_variant': variant,
            'probe_stress_tags': self._stress_tags_and_score(variant_sample)[0],
            'probe_family': 'mmmu_encoder_probe',
        }
        if variant != 'original':
            for message in variant_sample.input:
                if isinstance(message.content, list):
                    message.content = [
                        self._perturb_content_image(content, variant)
                        if getattr(content, 'type', None) == 'image' else content for content in message.content
                    ]
        return variant_sample

    def _perturb_content_image(self, content: ContentImage, variant: str) -> ContentImage:
        try:
            from PIL import Image, ImageFilter

            image_bytes = self._decode_data_uri(content.image)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert('RGB')
                if variant == 'downsample':
                    width, height = image.size
                    image = image.resize((max(1, width // 2), max(1, height // 2)))
                    image = image.resize((width, height))
                    kwargs = {'quality': 80}
                elif variant == 'jpeg_low':
                    kwargs = {'quality': 35, 'optimize': True}
                elif variant == 'grayscale':
                    image = image.convert('L').convert('RGB')
                    kwargs = {'quality': 80}
                elif variant == 'blur':
                    image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
                    kwargs = {'quality': 80}
                else:
                    return content
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', **kwargs)
                return ContentImage(
                    image=bytes_to_base64(buffer.getvalue(), format='jpeg', add_header=True), detail=content.detail
                )
        except Exception as exc:
            logger.warning(f'Could not apply MMMU image perturbation {variant}: {exc}')
            return content

    @staticmethod
    def _decode_data_uri(data_uri: str) -> bytes:
        match = re.match(r'^data:image/[^;]+;base64,(.*)$', data_uri, re.DOTALL)
        return base64.b64decode(match.group(1) if match else data_uri)
