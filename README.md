# KB

## Dependencies and Installation

1. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
   
2. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.

## API

### Vectorize document

**path**: `/` \
**content-type**: multipart/form-data \
**params**:
- `file`: file to upload (bytes)
- `path`: path of the file where it's stored e.g. FS (serves as unique identifier)
- `metadata`: arbitrary metadata to save alongside the file (optional)

### Ask document

**path**: `/ask` \
**content-type**: application/json \
**params**:
- `question`: question to ask
- `targeted_files`: paths of the files on which to ask the question
- `messages`: message history of the chat

### Extract CCCEV

**path**: `/cccevExtract` \
**content-type**: multipart/form-data \
**params**:
- `file`: file from which to extract the requirements (bytes)
- `path`: path of the file where it's stored e.g. FS (serves as unique identifier)
- `first_page`: first page to extract from, starting at 1 (default: 1)
- `last_page`: last page to extract from (default: 9999)
- `type`: type of extraction. Must be one of:
   - `METHODOLOGY_ELIGIBILITY`
