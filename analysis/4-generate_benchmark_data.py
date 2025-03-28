#!/usr/bin/env python
"""
Verification Benchmark Generator

This script generates challenging test cases specifically designed to evaluate 
latent verification mechanisms using established benchmark datasets:

1. GSM8K - Grade School Math 8K problems requiring multi-step reasoning
2. BBH - BIG-Bench Hard tasks testing advanced reasoning capabilities
3. MMLU - Massive Multitask Language Understanding benchmark covering 57 subjects

Each test case includes:
- Input prompt
- Expected correct output 
- Common incorrect output patterns
- Difficulty rating
- Category and subcategory labels

The benchmarks are saved in JSON format for use with the evaluation suite.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import requests
import csv
import io
import zipfile
import tarfile
import hashlib
from pathlib import Path

class VerificationBenchmarkGenerator:
    """Generator for verification benchmark datasets using established benchmarks"""

    def __init__(self, output_dir: str = "verification_benchmarks", data_dir: str = "benchmark_data"):
        """Initialize the benchmark generator"""
        self.output_dir = output_dir
        self.data_dir = data_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Storage for benchmark sets
        self.benchmarks = {}

        # Dataset URLs and file paths
        self.dataset_info = {
            "gsm8k": {
                "url": "https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/",
                "files": {
                    "train": "train.jsonl",
                    "test": "test.jsonl"
                },
                "local_path": os.path.join(data_dir, "gsm8k")
            },
            "bbh": {
                "url": "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip",
                "local_path": os.path.join(data_dir, "bbh")
            },
            "mmlu": {
                "url": "https://github.com/hendrycks/test/archive/refs/heads/master.zip",
                "local_path": os.path.join(data_dir, "mmlu")
            }
        }

    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download a dataset if it's not already available locally

        Args:
            dataset_name: Name of the dataset to download (gsm8k, bbh, or mmlu)

        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_name not in self.dataset_info:
            print(f"Unknown dataset: {dataset_name}")
            return False

        dataset = self.dataset_info[dataset_name]
        local_path = dataset["local_path"]

        # Check if the dataset already exists
        if os.path.exists(local_path) and len(os.listdir(local_path)) > 0:
            print(f"Dataset {dataset_name} already exists at {local_path}")
            return True

        os.makedirs(local_path, exist_ok=True)

        # Download and extract based on dataset type
        if dataset_name == "gsm8k":
            return self._download_gsm8k()
        elif dataset_name == "bbh":
            return self._download_and_extract_zip(dataset["url"], local_path)
        elif dataset_name == "mmlu":
            return self._download_and_extract_zip(dataset["url"], local_path)

        return False

    def _download_gsm8k(self) -> bool:
        """Download GSM8K dataset"""
        dataset = self.dataset_info["gsm8k"]
        local_path = dataset["local_path"]
        base_url = dataset["url"]

        os.makedirs(local_path, exist_ok=True)

        # Download train and test files
        for split, filename in dataset["files"].items():
            url = base_url + filename
            output_path = os.path.join(local_path, filename)

            if os.path.exists(output_path):
                print(f"File already exists: {output_path}")
                continue

            try:
                print(f"Downloading {url} to {output_path}")
                response = requests.get(url)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return False

        return True

    def _download_and_extract_zip(self, url: str, extract_path: str) -> bool:
        """Download and extract a ZIP file"""
        try:
            print(f"Downloading {url}")
            response = requests.get(url)
            response.raise_for_status()

            # Create a temporary file
            zip_path = os.path.join(self.data_dir, "temp.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)

            # Extract the ZIP file
            print(f"Extracting to {extract_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # Remove the temporary file
            os.remove(zip_path)
            print("Downloaded and extracted successfully")
            return True
        except Exception as e:
            print(f"Error downloading or extracting: {e}")
            return False

    def _process_gsm8k_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a GSM8K example into benchmark format"""
        # Extract the problem and solution
        question = example.get("question", "")
        answer = example.get("answer", "")

        # Extract the final answer from the solution
        final_answer = ""
        solution_lines = answer.strip().split("\n")
        if solution_lines and "####" in solution_lines[-1]:
            final_answer = solution_lines[-1].split("####")[1].strip()

        # Generate incorrect answers (simplified approach)
        if final_answer and final_answer.replace(',', '').replace('$', '').replace(' ', '').isdigit():
            # For numerical answers, modify slightly
            num = int(final_answer.replace(',', '').replace('$', '').replace(' ', ''))
            incorrect_answer = str(num + random.randint(1, max(1, num // 10)))

            # Format like the original
            if '$' in final_answer:
                incorrect_answer = f"${incorrect_answer}"
            if ',' in final_answer:
                # Add commas for thousands
                incorrect_answer = "{:,}".format(int(incorrect_answer.replace(',', '').replace('$', '')))
                if '$' in final_answer:
                    incorrect_answer = f"${incorrect_answer}"
        else:
            # For non-numerical answers, just append "incorrect" as placeholder
            incorrect_answer = final_answer + " (incorrect answer)"

        return {
            "prompt": question,
            "correct": final_answer if final_answer else answer,
            "incorrect": incorrect_answer,
            "difficulty": "medium",  # GSM8K problems are generally medium difficulty
            "category": "gsm8k",
            "subcategory": "math_word_problem",
            "full_solution": answer  # Keep the full solution for reference
        }

    def generate_gsm8k_benchmarks(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate benchmarks from GSM8K dataset

        This tests the model's ability to solve grade school math problems
        """
        print(f"Generating {num_samples} GSM8K benchmarks...")

        # Try to download the dataset if it doesn't exist
        if not os.path.exists(self.dataset_info["gsm8k"]["local_path"]):
            if not self.download_dataset("gsm8k"):
                print("Failed to download GSM8K dataset. Using small sample data instead.")
                return self._generate_sample_gsm8k(num_samples)

        benchmarks = []
        gsm8k_path = self.dataset_info["gsm8k"]["local_path"]

        # Load test and train data
        for split, filename in self.dataset_info["gsm8k"]["files"].items():
            filepath = os.path.join(gsm8k_path, filename)

            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    examples = [json.loads(line) for line in f]

                print(f"Loaded {len(examples)} examples from {filepath}")

                # Process each example
                for example in examples:
                    benchmark = self._process_gsm8k_example(example)
                    benchmarks.append(benchmark)

                    # Break if we have enough samples
                    if len(benchmarks) >= num_samples:
                        break

                if len(benchmarks) >= num_samples:
                    break
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        # If we still don't have enough, use the sample data
        if len(benchmarks) < num_samples:
            sample_benchmarks = self._generate_sample_gsm8k(num_samples - len(benchmarks))
            benchmarks.extend(sample_benchmarks)

        # Limit to requested number of samples
        if len(benchmarks) > num_samples:
            benchmarks = random.sample(benchmarks, num_samples)

        # Store and return
        self.benchmarks["gsm8k"] = benchmarks
        return benchmarks

    def _generate_sample_gsm8k(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate sample GSM8K benchmarks when dataset is not available"""
        print(f"Generating {num_samples} sample GSM8K benchmarks...")

        sample_problems = [
            {
                "question": "John has 5 apples. He buys 2 more apples and then gives 3 apples to his friend. How many apples does John have now?",
                "answer": "John starts with 5 apples.\nHe buys 2 more apples, so he has 5 + 2 = 7 apples.\nHe gives 3 apples to his friend, so he has 7 - 3 = 4 apples.\n#### 4"
            },
            {
                "question": "Sarah solved 35 math problems and 42 science problems this week. She solved 18 fewer problems last week. How many problems did she solve last week?",
                "answer": "This week, Sarah solved 35 + 42 = 77 problems.\nLast week, she solved 77 - 18 = 59 problems.\n#### 59"
            },
            {
                "question": "A store sells notebooks for $2.50 each. If you buy 3 or more notebooks, you get a 20% discount on all notebooks. How much will you pay for 5 notebooks?",
                "answer": "The regular price for 5 notebooks is 5 × $2.50 = $12.50.\nSince I'm buying more than 3 notebooks, I get a 20% discount.\nThe discount amount is 20% of $12.50 = 0.2 × $12.50 = $2.50.\nThe discounted price is $12.50 - $2.50 = $10.00.\n#### $10.00"
            },
            {
                "question": "James wants to tile a rectangular floor that is 12 feet by 8 feet. Each tile covers 2 square feet and costs $3. How much will it cost to tile the floor?",
                "answer": "The area of the floor is 12 × 8 = 96 square feet.\nEach tile covers 2 square feet, so James needs 96 ÷ 2 = 48 tiles.\nEach tile costs $3, so the total cost is 48 × $3 = $144.\n#### $144"
            },
            {
                "question": "A bakery makes 200 cookies. They sell 40% of the cookies in the morning and 35% of the remaining cookies in the afternoon. How many cookies are left?",
                "answer": "In the morning, they sell 40% of 200 = 0.4 × 200 = 80 cookies.\nAfter the morning, they have 200 - 80 = 120 cookies left.\nIn the afternoon, they sell 35% of the remaining 120 cookies = 0.35 × 120 = 42 cookies.\nAfter both morning and afternoon, they have 120 - 42 = 78 cookies left.\n#### 78"
            },
            {
                "question": "Mark reads 20 pages of his book every day. If the book has 380 pages and he has already read 60 pages, how many more days will it take him to finish the book?",
                "answer": "The book has 380 pages in total.\nMark has already read 60 pages.\nSo he has 380 - 60 = 320 pages left to read.\nHe reads 20 pages every day.\nSo it will take him 320 ÷ 20 = 16 days to finish the book.\n#### 16"
            },
            {
                "question": "In a class of 32 students, 3/8 are boys. How many girls are in the class?",
                "answer": "There are 32 students in the class.\n3/8 of the students are boys.\nSo the number of boys is 3/8 × 32 = 3 × 4 = 12.\nThe number of girls is 32 - 12 = 20.\n#### 20"
            },
            {
                "question": "A train travels 240 miles at a constant speed. If the journey takes 4 hours, what is the speed of the train in miles per hour?",
                "answer": "The train travels 240 miles in 4 hours.\nThe speed is distance divided by time.\nSpeed = 240 miles ÷ 4 hours = 60 miles per hour.\n#### 60"
            },
            {
                "question": "Tony has $25. He spends 2/5 of his money on lunch and then buys a book for $10. How much money does he have left?",
                "answer": "Tony has $25.\nHe spends 2/5 of his money on lunch, which is 2/5 × $25 = $10.\nAfter lunch, he has $25 - $10 = $15 left.\nThen he buys a book for $10.\nSo he has $15 - $10 = $5 left.\n#### $5"
            },
            {
                "question": "A recipe requires 2 3/4 cups of flour. If Maria wants to make 1/3 of the recipe, how many cups of flour will she need?",
                "answer": "The recipe requires 2 3/4 = 2.75 cups of flour.\nIf Maria makes 1/3 of the recipe, she will need 1/3 × 2.75 = 2.75 ÷ 3 = 0.916... cups of flour.\nRounded to the nearest standard fraction, that's 11/12 or about 0.92 cups.\n#### 11/12"
            }
        ]

        benchmarks = []
        # Use the available samples
        for i in range(min(num_samples, len(sample_problems))):
            benchmark = self._process_gsm8k_example(sample_problems[i])
            benchmarks.append(benchmark)

        # If we need more, duplicate with variations
        while len(benchmarks) < num_samples:
            # Choose a random sample and modify the numbers slightly
            sample = random.choice(sample_problems)
            modified_sample = dict(sample)

            # This is a simple modification approach
            modified_sample["question"] = sample["question"].replace("5", str(random.randint(3, 9)))

            benchmark = self._process_gsm8k_example(modified_sample)
            benchmarks.append(benchmark)

        return benchmarks

    def _process_bbh_example(self, task: str, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a BBH example into benchmark format"""
        input_text = example.get("input", "")
        target = example.get("target", "")

        # Generate incorrect answer (simplified)
        if task in ["dyck_languages", "formal_fallacies"]:
            # For binary tasks, flip the answer
            if target.lower() in ["yes", "true", "valid"]:
                incorrect = "No"
            else:
                incorrect = "Yes"
        elif task in ["logical_deduction", "hyperbaton", "movie_recommendation"]:
            # For multiple choice, choose a different option
            options = ["A", "B", "C", "D", "E"]
            if target in options:
                incorrect_options = [opt for opt in options if opt != target and opt in input_text]
                incorrect = random.choice(incorrect_options) if incorrect_options else "A"
            else:
                incorrect = "A"  # Default fallback
        else:
            # For other tasks, modify the target slightly
            incorrect = target + " (incorrect)"

        # Determine difficulty
        if task in ["logical_deduction", "object_counting", "date_understanding"]:
            difficulty = "medium"
        elif task in ["multistep_arithmetic", "formal_fallacies", "tracking_shuffled_objects"]:
            difficulty = "hard"
        else:
            difficulty = "medium"  # Default

        return {
            "prompt": input_text,
            "correct": target,
            "incorrect": incorrect,
            "difficulty": difficulty,
            "category": "bbh",
            "subcategory": task
        }

    def generate_bbh_benchmarks(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate benchmarks from BIG-Bench Hard dataset

        This tests the model's reasoning capabilities across different tasks
        """
        print(f"Generating {num_samples} BBH benchmarks...")

        # Try to download the dataset if it doesn't exist
        if not os.path.exists(self.dataset_info["bbh"]["local_path"]):
            if not self.download_dataset("bbh"):
                print("Failed to download BBH dataset. Using sample data instead.")
                return self._generate_sample_bbh(num_samples)

        benchmarks = []
        bbh_path = self.dataset_info["bbh"]["local_path"]

        # Find the actual BBH directory after extraction
        bbh_dirs = [d for d in os.listdir(bbh_path) if "BIG-Bench-Hard" in d]
        if not bbh_dirs:
            print(f"BBH directory not found in {bbh_path}")
            return self._generate_sample_bbh(num_samples)

        bbh_dir = os.path.join(bbh_path, bbh_dirs[0], "bbh")

        # BBH tasks
        bbh_tasks = [
            "boolean_expressions", "causal_judgement", "date_understanding",
            "disambiguation_qa", "dyck_languages", "formal_fallacies",
            "geometric_shapes", "hyperbaton", "logical_deduction", "movie_recommendation",
            "multistep_arithmetic", "navigate", "object_counting", "penguins_in_a_table",
            "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
            "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects",
            "web_of_lies", "word_sorting"
        ]

        # Shuffle tasks to get a mix
        random.shuffle(bbh_tasks)

        for task in bbh_tasks:
            task_dir = os.path.join(bbh_dir, task)

            if not os.path.exists(task_dir):
                print(f"Task directory not found: {task_dir}")
                continue

            # Look for the task file
            task_files = [f for f in os.listdir(task_dir) if f.endswith(".json")]
            if not task_files:
                print(f"No task files found in {task_dir}")
                continue

            # Load the task file
            task_file = os.path.join(task_dir, task_files[0])
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                examples = task_data.get("examples", [])
                print(f"Loaded {len(examples)} examples from {task_file}")

                # Process examples
                task_benchmarks = []
                for example in examples:
                    benchmark = self._process_bbh_example(task, example)
                    task_benchmarks.append(benchmark)

                # Take a sample from this task
                samples_per_task = max(1, min(20, num_samples // len(bbh_tasks)))
                if len(task_benchmarks) > samples_per_task:
                    task_benchmarks = random.sample(task_benchmarks, samples_per_task)

                benchmarks.extend(task_benchmarks)

                # Break if we have enough samples
                if len(benchmarks) >= num_samples:
                    break
            except Exception as e:
                print(f"Error processing {task_file}: {e}")

        # If we still don't have enough, use the sample data
        if len(benchmarks) < num_samples:
            sample_benchmarks = self._generate_sample_bbh(num_samples - len(benchmarks))
            benchmarks.extend(sample_benchmarks)

        # Limit to requested number of samples
        if len(benchmarks) > num_samples:
            benchmarks = random.sample(benchmarks, num_samples)

        # Store and return
        self.benchmarks["bbh"] = benchmarks
        return benchmarks

    def _generate_sample_bbh(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate sample BBH benchmarks when dataset is not available"""
        print(f"Generating {num_samples} sample BBH benchmarks...")

        sample_tasks = {
            "boolean_expressions": [
                {
                    "input": "Solve the boolean expression: (NOT (True OR False)) AND (True AND NOT False)",
                    "target": "False"
                }
            ],
            "logical_deduction": [
                {
                    "input": "Iris, Jasmine, and Lily are three sisters. Their surnames are Miller, Watson, and Smith, but not necessarily in that order. The sister with the surname Miller is the oldest. Jasmine is older than the sister with the surname Smith. Lily is not the sister with the surname Watson. Lily is the youngest. Based on this information, what is Iris's surname?\nA. Miller\nB. Smith\nC. Watson",
                    "target": "A"
                }
            ],
            "multistep_arithmetic": [
                {
                    "input": "If 9 people meet and each person shakes hands exactly once with each of the others, how many handshakes take place?",
                    "target": "36"
                }
            ],
            "date_understanding": [
                {
                    "input": "Today is 27 February 2021. What is the date 10 days ago in MM/DD/YYYY?",
                    "target": "02/17/2021"
                }
            ],
            "tracking_shuffled_objects": [
                {
                    "input": "I have 3 cards, labeled A, B, and C. I place them in an ordered deck A,B,C from top to bottom. I perform the following operations:\n- Swap the top card and the bottom card.\n- Swap the middle card and the bottom card.\n- Swap the top card and the middle card.\nWhat is the ordering of the deck from top to bottom?",
                    "target": "B,A,C"
                }
            ]
        }

        benchmarks = []

        # Use each sample task at least once
        for task, examples in sample_tasks.items():
            for example in examples:
                benchmark = self._process_bbh_example(task, example)
                benchmarks.append(benchmark)

                if len(benchmarks) >= num_samples:
                    break

            if len(benchmarks) >= num_samples:
                break

        # If we need more samples, duplicate with variations
        while len(benchmarks) < num_samples:
            task = random.choice(list(sample_tasks.keys()))
            example = random.choice(sample_tasks[task])

            # Simple variation
            modified_example = dict(example)
            if task == "multistep_arithmetic":
                # Modify numbers in the arithmetic problem
                input_text = example["input"]
                for i in range(1, 10):
                    if str(i) in input_text:
                        input_text = input_text.replace(str(i), str(random.randint(1, 9)), 1)
                        break
                modified_example["input"] = input_text

            benchmark = self._process_bbh_example(task, modified_example)
            benchmarks.append(benchmark)

        return benchmarks

    def _process_mmlu_example(self, subject: str, example: List[str]) -> Dict[str, Any]:
        """Process an MMLU example into benchmark format"""
        # MMLU format is typically: question, A, B, C, D, answer
        if len(example) < 6:
            return None

        question = example[0]
        options = example[1:5]
        correct_answer = example[5]

        # Format prompt with options
        prompt = f"{question}\n"
        for i, option in enumerate(options):
            prompt += f"{chr(65+i)}. {option}\n"

        # Map answer to option text
        if correct_answer in "ABCD":
            correct_index = ord(correct_answer) - ord('A')
            if 0 <= correct_index < len(options):
                correct = options[correct_index]
            else:
                correct = correct_answer
        else:
            correct = correct_answer

        # Generate incorrect answer
        if correct_answer in "ABCD":
            incorrect_options = [opt for i, opt in enumerate(options) if chr(65+i) != correct_answer]
            incorrect = random.choice(incorrect_options) if incorrect_options else options[0]
        else:
            incorrect = random.choice(options) if options else "Incorrect answer"

        # Determine difficulty based on subject
        stem_subjects = ["abstract_algebra", "calculus", "college_chemistry", "college_physics", "high_school_physics", "high_school_statistics"]
        professional_subjects = ["medical_genetics", "professional_medicine", "professional_law", "international_law", "clinical_knowledge"]

        if subject in stem_subjects:
            difficulty = "hard"
        elif subject in professional_subjects:
            difficulty = "very_hard"
        else:
            difficulty = "medium"

        return {
            "prompt": prompt.strip(),
            "correct": correct,
            "incorrect": incorrect,
            "difficulty": difficulty,
            "category": "mmlu",
            "subcategory": subject,
            "correct_option": correct_answer  # Store the original correct option letter
        }

    def generate_mmlu_benchmarks(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate benchmarks from MMLU dataset

        This tests the model's knowledge across 57 subjects
        """
        print(f"Generating {num_samples} MMLU benchmarks...")

        # Try to download the dataset if it doesn't exist
        if not os.path.exists(self.dataset_info["mmlu"]["local_path"]):
            if not self.download_dataset("mmlu"):
                print("Failed to download MMLU dataset. Using sample data instead.")
                return self._generate_sample_mmlu(num_samples)

        benchmarks = []
        mmlu_path = self.dataset_info["mmlu"]["local_path"]

        # Find the actual MMLU directory after extraction
        mmlu_dirs = [d for d in os.listdir(mmlu_path) if "test" in d.lower()]
        if not mmlu_dirs:
            print(f"MMLU directory not found in {mmlu_path}")
            return self._generate_sample_mmlu(num_samples)

        mmlu_dir = os.path.join(mmlu_path, mmlu_dirs[0], "data")

        # Get all subject directories
        if not os.path.exists(mmlu_dir):
            print(f"MMLU data directory not found: {mmlu_dir}")
            return self._generate_sample_mmlu(num_samples)

        # Get all unique subjects
        subjects = set()
        for file in os.listdir(mmlu_dir):
            if file.endswith("_test.csv"):
                subject = file.replace("_test.csv", "")
                subjects.add(subject)

        # Shuffle subjects to get a mix
        subjects = list(subjects)
        random.shuffle(subjects)

        # Determine number of samples per subject
        samples_per_subject = max(1, min(5, num_samples // len(subjects)))

        for subject in subjects:
            test_file = os.path.join(mmlu_dir, f"{subject}_test.csv")

            if not os.path.exists(test_file):
                continue

            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    examples = list(reader)

                print(f"Loaded {len(examples)} examples from {test_file}")

                # Process examples
                subject_benchmarks = []
                for example in examples:
                    benchmark = self._process_mmlu_example(subject, example)
                    if benchmark:
                        subject_benchmarks.append(benchmark)

                # Take a sample from this subject
                if len(subject_benchmarks) > samples_per_subject:
                    subject_benchmarks = random.sample(subject_benchmarks, samples_per_subject)

                benchmarks.extend(subject_benchmarks)

                # Break if we have enough samples
                if len(benchmarks) >= num_samples:
                    break
            except Exception as e:
                print(f"Error processing {test_file}: {e}")

        # If we still don't have enough, use the sample data
        if len(benchmarks) < num_samples:
            sample_benchmarks = self._generate_sample_mmlu(num_samples - len(benchmarks))
            benchmarks.extend(sample_benchmarks)

        # Limit to requested number of samples
        if len(benchmarks) > num_samples:
            benchmarks = random.sample(benchmarks, num_samples)

        # Store and return
        self.benchmarks["mmlu"] = benchmarks
        return benchmarks

    def _generate_sample_mmlu(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate sample MMLU benchmarks when dataset is not available"""
        print(f"Generating {num_samples} sample MMLU benchmarks...")

        sample_questions = [
            {
                "subject": "high_school_mathematics",
                "example": [
                    "What is the solution to the equation 3x + 7 = 22?",
                    "x = 5",
                    "x = 15/3",
                    "x = 29/3",
                    "None of the above",
                    "A"
                ]
            },
            {
                "subject": "high_school_biology",
                "example": [
                    "Which of the following is NOT a function of the liver?",
                    "Production of bile",
                    "Storage of glycogen",
                    "Production of insulin",
                    "Detoxification of blood",
                    "C"
                ]
            },
            {
                "subject": "high_school_chemistry",
                "example": [
                    "What is the chemical symbol for gold?",
                    "Go",
                    "Gd",
                    "Gl",
                    "Au",
                    "D"
                ]
            },
            {
                "subject": "high_school_physics",
                "example": [
                    "Which of the following is the unit of electric current?",
                    "Volt",
                    "Ampere",
                    "Ohm",
                    "Watt",
                    "B"
                ]
            },
            {
                "subject": "high_school_geography",
                "example": [
                    "Which of the following countries is NOT in Africa?",
                    "Ethiopia",
                    "Kenya",
                    "Cambodia",
                    "Nigeria",
                    "C"
                ]
            },
            {
                "subject": "high_school_history",
                "example": [
                    "The Treaty of Versailles was signed at the end of which war?",
                    "World War I",
                    "World War II",
                    "The Cold War",
                    "The American Civil War",
                    "A"
                ]
            },
            {
                "subject": "high_school_literature",
                "example": [
                    "Who wrote 'Pride and Prejudice'?",
                    "Jane Austen",
                    "Charlotte Brontë",
                    "Virginia Woolf",
                    "George Eliot",
                    "A"
                ]
            },
            {
                "subject": "professional_medicine",
                "example": [
                    "Which of the following is NOT a symptom typically associated with acute appendicitis?",
                    "Right lower quadrant pain",
                    "Rebound tenderness",
                    "Nausea and vomiting",
                    "Rash on the abdomen",
                    "D"
                ]
            },
            {
                "subject": "professional_law",
                "example": [
                    "In U.S. contract law, what is the term for a promise that is exchanged for another promise?",
                    "Consideration",
                    "Bilateral contract",
                    "Promissory estoppel",
                    "Unilateral contract",
                    "B"
                ]
            },
            {
                "subject": "college_computer_science",
                "example": [
                    "Which data structure operates on a First-In-First-Out (FIFO) principle?",
                    "Stack",
                    "Queue",
                    "Tree",
                    "Heap",
                    "B"
                ]
            }
        ]

        benchmarks = []

        # Use each sample at least once
        for sample in sample_questions:
            benchmark = self._process_mmlu_example(sample["subject"], sample["example"])
            if benchmark:
                benchmarks.append(benchmark)

            if len(benchmarks) >= num_samples:
                break

        # If we need more, duplicate with variations
        while len(benchmarks) < num_samples:
            sample = random.choice(sample_questions)

            # Create a variation (in this simple case, just use the original)
            benchmark = self._process_mmlu_example(sample["subject"], sample["example"])
            if benchmark:
                benchmarks.append(benchmark)

        return benchmarks

    def generate_all_benchmarks(self, samples_per_category: int = 100):
        """
        Generate benchmarks for all categories

        Args:
            samples_per_category: Number of samples to generate per category
        """
        self.generate_gsm8k_benchmarks(samples_per_category)
        self.generate_bbh_benchmarks(samples_per_category)
        self.generate_mmlu_benchmarks(samples_per_category)

        # Save all benchmarks to file
        self.save_benchmarks()

        return self.benchmarks

    def save_benchmarks(self, filename: str = None):
        """
        Save benchmarks to JSON file

        Args:
            filename: Filename to save to (if None, will use 'verification_benchmarks.json')
        """
        if filename is None:
            filename = os.path.join(self.output_dir, "verification_benchmarks.json")

        # Count statistics
        stats = {
            "total": 0,
            "by_category": {},
            "by_difficulty": {
                "easy": 0,
                "medium": 0,
                "hard": 0,
                "very_hard": 0
            }
        }

        for category, benchmarks in self.benchmarks.items():
            stats["by_category"][category] = len(benchmarks)
            stats["total"] += len(benchmarks)

            # Count by difficulty
            for benchmark in benchmarks:
                difficulty = benchmark.get("difficulty", "medium")
                stats["by_difficulty"][difficulty] += 1

        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                "stats": stats,
                "benchmarks": self.benchmarks
            }, f, indent=2)

        print(f"Saved {stats['total']} benchmarks to {filename}")
        print(f"Statistics: {json.dumps(stats, indent=2)}")

    def load_benchmarks(self, filename: str = None):
        """
        Load benchmarks from JSON file

        Args:
            filename: Filename to load from (if None, will use 'verification_benchmarks.json')
        """
        if filename is None:
            filename = os.path.join(self.output_dir, "verification_benchmarks.json")

        with open(filename, 'r') as f:
            data = json.load(f)
            self.benchmarks = data["benchmarks"]

        return self.benchmarks

def main():
    """Main function to run the benchmark generator"""
    parser = argparse.ArgumentParser(description="Generate verification benchmarks using established datasets")
    parser.add_argument("--output_dir", type=str, default="verification_benchmarks",
                        help="Directory to save benchmark files")
    parser.add_argument("--data_dir", type=str, default="benchmark_data",
                        help="Directory to store downloaded datasets")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples per benchmark category")
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=["gsm8k", "bbh", "mmlu", "all"],
                        default=["all"],
                        help="Categories of benchmarks to generate")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download datasets without generating benchmarks")

    args = parser.parse_args()

    # Create generator
    generator = VerificationBenchmarkGenerator(output_dir=args.output_dir, data_dir=args.data_dir)

    if args.download_only:
        # Just download the datasets
        if "all" in args.categories:
            for dataset in ["gsm8k", "bbh", "mmlu"]:
                print(f"Downloading {dataset} dataset...")
                generator.download_dataset(dataset)
        else:
            for dataset in args.categories:
                print(f"Downloading {dataset} dataset...")
                generator.download_dataset(dataset)
        return

    # Generate benchmarks
    if "all" in args.categories:
        generator.generate_all_benchmarks(args.samples)
    else:
        # Generate requested categories
        if "gsm8k" in args.categories:
            generator.generate_gsm8k_benchmarks(args.samples)
        if "bbh" in args.categories:
            generator.generate_bbh_benchmarks(args.samples)
        if "mmlu" in args.categories:
            generator.generate_mmlu_benchmarks(args.samples)

        # Save benchmarks
        generator.save_benchmarks()

if __name__ == "__main__":
    main()
