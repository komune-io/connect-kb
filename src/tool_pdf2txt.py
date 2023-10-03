from langchain.tools import BaseTool
from langchain.document_loaders import PyPDFium2Loader as PdfReader
import aspose.words as aw


class Pdf2TxtTool(BaseTool):
    name = "Pdf2Txt"
    description = "useful for when you need to extract the content of one page of a pdf as plain text"

    def _run(self, input_file_path: str, page_number: int):
        reader = PdfReader(input_file_path)
        # pages = reader.load()
        # return '\n'.join(map(lambda page: page.page_content, pages))
        pages = reader.load()
        return {
            "page_content": pages[page_number - 1].page_content,
            "page_number": page_number,
            "total_pages": len(pages)
        }


class Pdf2MdTool(BaseTool):
    name = "Pdf2Md"
    description = "useful for when you need to convert a pdf file into a markdown file"

    def _run(self, input_file_path: str, page_number: int, output_file_path: str):
        doc = aw.Document(input_file_path)
        doc.save(output_file_path)
