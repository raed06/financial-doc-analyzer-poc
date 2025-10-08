import os
import logging
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles document loading and processing"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF files"""
        documents = []

        try:
            # Using pdfplumber for comples PDFs
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata = {
                                    "source": os.path.basename(file_path),
                                    "page": page_num + 1,
                                    "type": "pdf"
                                }
                            )
                        )
            logger.info(f"Loaded PDF with pdfplumber: {file_path}")
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")

            # Using PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata = {
                                        "source": os.path.basename(file_path),
                                        "page": page_num + 1,
                                        "type": "pdf"
                                    }
                                )
                            )
                logger.info(f"Loaded PDF with PyPDF2: {file_path}")
            except Exception as e2:
                logger.warning(f"Failed to load PDF: {e2}")
                raise
        
        return documents
    
    def load_csv(self, file_path: str) -> List[Document]:
        """Load and process CSV files"""
        documents = []

        try:
            df = pd.read_csv(file_path)

            # Create a summary document
            summary = f"CSV File: {os.path.basename(file_path)}\n"
            summary += f"Columns: {', '.join(df.columns)}\n"
            summary += f"Total Rows: {len(df)}\n\n"
            summary += f"Data Summary:\n{df.describe().to_string()}\n\n"

            documents.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "cvs",
                        "rows": len(df),
                        "columns": len(df.columns)
                    }
                )
            )

            # Convert rows to text documents (in chunks)
            chunk_size = 50
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                text = f"Data from {os.path.basename(file_path)} (rows {i+1}-{i+len(chunk)}):\n"
                text += chunk.to_string(index=False)

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "type": "csv_chunk",
                            "chunk_start": i,
                            "chunk_end": i + len(chunk)
                        }
                    )
                )
            
            logger.info(f"Loaded CSV: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            raise

    def load_file(self, file_path: str) -> List[Document]:
        """Load file based on extension"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            docs = self.load_pdf(file_path)
        elif ext == '.csv':
            docs = self.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return self.process_documents(docs)