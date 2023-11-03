from io import BytesIO

import tiktoken
from PyPDF2 import PdfReader


class PdfSplitter:
    token_limit: int
    llm_model_name: str

    def __init__(self, token_limit: int, llm_model_name: str):
        self.token_limit = token_limit
        self.llm_model_name = llm_model_name

    def group_pages(self, file_content: BytesIO, from_page: int, to_page: int) -> list[str]:
        pdf_reader = PdfReader(file_content)
        token_encoder = tiktoken.encoding_for_model(self.llm_model_name)

        page_groups = [""]
        current_group_idx = 0
        current_group_size = 0
        current_page_number = from_page

        for page in pdf_reader.pages[from_page-1:to_page]:
            page_text = f"Page {current_page_number}: {page.extract_text()}\n---\n"
            page_size = len(token_encoder.encode(page_text))

            if (current_group_size + page_size) > self.token_limit:
                current_group_idx += 1
                current_group_size = page_size
                page_groups.append(page_text)
            else:
                current_group_size += page_size
                page_groups[current_group_idx] += page_text

            current_page_number += 1

        return page_groups
