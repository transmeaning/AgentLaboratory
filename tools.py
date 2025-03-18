from utils import *

import time
import arxiv
import os, re
import io, sys
import numpy as np
import concurrent.futures
from pypdf import PdfReader
from datasets import load_dataset
from psutil._common import bytes2human
from datasets import load_dataset_builder
from semanticscholar import SemanticScholar
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import traceback
import concurrent.futures
import logging
import tempfile
from PyPDF2 import PdfReader


class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant huggingface datasets
        :param like_thr:
        :param dwn_thr:
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            # Get likes and downloads, handling None values
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            # Check if likes and downloads meet the thresholds
            if likes >= self.like_thr and downloads >= self.dwn_thr:
                # Check if the description is a non-empty string
                description = item['description']
                if isinstance(description, str) and description.strip():
                    # Collect the data
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit the constructor

        # Filter the datasets using the collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update descriptions, likes, and downloads
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query, weighted by likes and downloads.
        :param query: The search query string.
        :param N: The number of results to return.
        :param sim_w: Weight for cosine similarity.
        :param like_w: Weight for likes.
        :param dwn_w: Weight for downloads.
        :return: List of top N dataset items.
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        # Normalize cosine similarities
        cosine_similarities_norm = self._normalize(cosine_similarities)
        # Compute final scores
        final_scores = (
                sim_w * cosine_similarities_norm +
                like_w * self.likes_norm +
                dwn_w * self.downloads_norm
        )
        # Get top N indices
        top_indices = final_scores.argsort()[-N:][::-1]
        # Convert indices to Python ints
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]
        # check if dataset has a test & train set
        has_test_set = list()
        has_train_set = list()
        ds_size_info = list()
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception as e:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue
            # Print number of examples for
            has_test, has_train = "test" in dbuilder.splits, "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)
            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None
            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples
            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))
        for _i in range(len(top_datasets)):
            top_datasets[_i]["has_test_set"] = has_test_set[_i]
            top_datasets[_i]["has_train_set"] = has_train_set[_i]
            top_datasets[_i]["test_download_size"] = ds_size_info[_i][0]
            top_datasets[_i]["test_element_size"] = ds_size_info[_i][1]
            top_datasets[_i]["train_download_size"] = ds_size_info[_i][2]
            top_datasets[_i]["train_element_size"] = ds_size_info[_i][3]
        return top_datasets

    def results_str(self, results):
        """
        Provide results as list of results in human-readable format.
        :param results: (list(dict)) list of results from search
        :return: (list(str)) list of results in human-readable format
        """
        result_strs = list()
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs


class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        paper_sums = list()
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for _i in range(len(results)):
            paper_sum = f'Title: {results[_i].title}\n'
            paper_sum += f'Abstract: {results[_i].abstract}\n'
            paper_sum += f'Citations: {results[_i].citationCount}\n'
            paper_sum += f'Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n'
            paper_sum += f'Venue: {results[_i].venue}\n'
            paper_sum += f'Paper ID: {results[_i].externalIds["DOI"]}\n'
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        pass


class ArxivSearch:
    def __init__(self):
        self.client = arxiv.Client()
        print("ArXiv search initialized")
        logging.info("ArXiv search initialized")

    def find_papers_by_str(self, search_str, N=5):
        """
        Find papers on ArXiv by search string.
        
        Args:
            search_str: The search string
            N: Maximum number of papers to return
            
        Returns:
            List[Dict]: A list of paper summaries
        """
        print(f"Searching ArXiv for: {search_str}")
        logging.info(f"Searching ArXiv for: {search_str}")
        
        search = arxiv.Search(
            query=search_str,
            max_results=N,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in self.client.results(search):
            paper_summary = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "arxiv_id": result.entry_id.split("/")[-1]
            }
            results.append(paper_summary)
            print(f"Found paper: {result.title} (ID: {paper_summary['arxiv_id']})")
            logging.info(f"Found paper: {result.title} (ID: {paper_summary['arxiv_id']})")
        
        print(f"Found {len(results)} papers on ArXiv")
        logging.info(f"Found {len(results)} papers on ArXiv")
        return results

    def retrieve_full_paper_text(self, arxiv_id):
        """
        Retrieve the full text of a paper from ArXiv.
        
        Args:
            arxiv_id: The ArXiv ID of the paper
            
        Returns:
            str: The full text of the paper
        """
        print(f"Retrieving full text for paper with ID: {arxiv_id}")
        logging.info(f"Retrieving full text for paper with ID: {arxiv_id}")
        
        search = arxiv.Search(
            id_list=[arxiv_id],
            max_results=1
        )
        
        try:
            result = next(self.client.results(search))
            
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the PDF
                pdf_path = os.path.join(temp_dir, f"{arxiv_id}.pdf")
                result.download_pdf(filename=pdf_path)
                
                # Extract text from the PDF
                text = extract_text_from_pdf(pdf_path)
                
                print(f"Successfully retrieved full text for paper with ID: {arxiv_id} ({len(text)} characters)")
                logging.info(f"Successfully retrieved full text for paper with ID: {arxiv_id} ({len(text)} characters)")
                return text
        except Exception as e:
            print(f"Error retrieving full text for paper with ID: {arxiv_id}: {e}")
            logging.error(f"Error retrieving full text for paper with ID: {arxiv_id}: {e}")
            return f"Error retrieving paper: {e}"

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: The extracted text
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {page_num} ---\n{page_text}\n\n"
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return f"Error extracting text from PDF: {e}"

"""
import multiprocessing
import sys
import io
import traceback

def execute_code(code_str, timeout=180):
    if "load_dataset('pubmed" in code_str:
        return "pubmed Download took way too long. Program terminated"

    def run_code(queue):
        # Redirect stdout to capture print outputs
        output_capture = io.StringIO()
        sys.stdout = output_capture

        try:
            exec_globals = {}
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)
        finally:
            # Put the output in the queue
            queue.put(output_capture.getvalue())
            # Restore stdout
            sys.stdout = sys.__stdout__

    # Create a multiprocessing Queue to capture the output
    queue = multiprocessing.Queue()
    # Create a new Process
    process = multiprocessing.Process(target=run_code, args=(queue,))
    process.start()
    # Wait for the process to finish or timeout
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. You must reduce the time complexity of your code."
    else:
        # Retrieve the output from the queue
        output = queue.get()
        return output

"""

import io
import sys
import traceback
import concurrent.futures



import multiprocessing
import io
import sys
import traceback
import multiprocessing
import io
import sys
import traceback


def execute_code(code_str, timeout=60, MAX_LEN=1000):
    #print(code_str)

    # prevent plotting errors
    import matplotlib
    matplotlib.use('Agg')  # Use the non-interactive Agg backend
    import matplotlib.pyplot as plt

    # Preventing execution of certain resource-intensive datasets
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."
    #print(code_str)
    # Capturing the output
    output_capture = io.StringIO()
    sys.stdout = output_capture

    # Create a new global context for exec
    exec_globals = globals()

    def run_code():
        try:
            # Executing the code in the global namespace
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)

    try:
        # Running code in a separate thread with a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_code)
            future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. You must reduce the time complexity of your code."
    except Exception as e:
        return f"[CODE EXECUTION ERROR]: {str(e)}"
    finally:
        # Restoring standard output
        sys.stdout = sys.__stdout__

    # Returning the captured output
    return output_capture.getvalue()[:MAX_LEN]



