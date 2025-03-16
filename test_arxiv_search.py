import unittest
import time
from tools import ArxivSearch
from unittest.mock import patch, call
import requests
import arxiv

class TestArxivSearch(unittest.TestCase):
    def setUp(self):
        self.arxiv_search = ArxivSearch()
        self.test_paper_id = "2401.00123"  # Use a known paper ID for testing

    def test_successful_paper_retrieval(self):
        """Test successful paper retrieval"""
        result = self.arxiv_search.retrieve_full_paper_text(self.test_paper_id)
        self.assertIsNotNone(result)
        self.assertNotEqual(result, "FAILED AFTER MAX RETRIES")
        self.assertNotEqual(result, "NO RESULTS FOUND")

    def test_invalid_paper_id(self):
        """Test handling of invalid paper ID"""
        with patch('arxiv.Client.results') as mock_results:
            mock_results.side_effect = arxiv.HTTPError("test_url", 1, 400)
            result = self.arxiv_search.retrieve_full_paper_text("invalid_id")
            self.assertEqual(result, "FAILED AFTER MAX RETRIES")

    @patch('requests.Session.get')
    @patch('time.sleep')  # Add mock for sleep to speed up tests
    def test_connection_error_retry(self, mock_sleep, mock_get):
        """Test retry mechanism for connection errors"""
        # Create a mock response for the successful case
        mock_response = type('MockResponse', (), {
            'content': b'test content',
            'raise_for_status': lambda: None
        })()
        
        # Configure mock to raise ConnectionError twice then succeed
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("First failure"),
            requests.exceptions.ConnectionError("Second failure"),
            mock_response
        ]
        
        # Mock both arxiv.Client and arxiv.Search
        with patch('arxiv.Client') as mock_client, \
             patch('arxiv.Search') as mock_search:
            # Create a mock paper with a PDF URL
            mock_paper = type('MockPaper', (), {'pdf_url': 'http://test.com/paper.pdf'})()
            
            # Configure the search mock to return a valid search object
            mock_search.return_value = type('MockSearch', (), {'id_list': [self.test_paper_id]})()
            
            # Configure the client mock to always return our paper
            mock_client_instance = mock_client.return_value
            mock_results = type('MockResults', (), {'__next__': lambda _: mock_paper})()
            mock_client_instance.results = lambda _: mock_results
            
            # Mock PdfReader to avoid actual PDF processing
            with patch('pypdf.PdfReader') as mock_reader:
                # Create a mock page that returns test content
                mock_page = type('MockPage', (), {'extract_text': lambda: 'Test content'})()
                mock_reader.return_value.pages = [mock_page]
                
                result = self.arxiv_search.retrieve_full_paper_text(self.test_paper_id)
                
                # Verify the correct number of retries occurred
                self.assertEqual(mock_get.call_count, 3)
                
                # Verify sleep was called with correct backoff times
                expected_calls = [
                    call(2.0),  # First retry after 2 seconds
                    call(4.0),  # Second retry after 4 seconds
                ]
                self.assertEqual(mock_sleep.call_args_list, expected_calls)
                
                # Verify we got the expected content
                self.assertIn('Test content', result)

    def test_cleanup_after_error(self):
        """Test that temporary files are cleaned up after errors"""
        import os
        pdf_filename = "downloaded-paper.pdf"
        
        # Force an error during PDF processing
        with patch('pypdf.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("PDF processing error")
            self.arxiv_search.retrieve_full_paper_text(self.test_paper_id)
            
            # Verify the temporary file was cleaned up
            self.assertFalse(os.path.exists(pdf_filename))

if __name__ == '__main__':
    unittest.main() 