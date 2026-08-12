import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_FILE = BASE_DIR / "chunks.jsonl"


DOCUMENT_TOPICS = {
    "irctc_eticket_faq.pdf": "e_ticket",
    "irctc_terms_conditions.pdf": "booking_rules",
    "irctc_cancellation_refund_rules.pdf": "cancellation_refund",
    "indian_railways_luggage_rules.pdf": "luggage",
    "indian_railways_concession_rules.pdf": "concessions",
    "irctc_tatkal_faq.pdf": "tatkal",
}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def load_documents():
    documents = []

    for filename, topic in DOCUMENT_TOPICS.items():

        path = DOCUMENTS_DIR / filename

        if not path.exists():
            print(f"WARNING: Missing {filename}")
            continue

        loader = PyPDFLoader(str(path))
        pages = loader.load()

        for page in pages:
            page.page_content = clean_text(
                page.page_content
            )

            page.metadata.update({
                "source": filename,
                "topic": topic,
                "authority": "IRCTC / Indian Railways",
                "document_type": "official_rules",
            })

        documents.extend(pages)

    return documents


def main():

    print("Loading official railway documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} pages."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            " ",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for index, chunk in enumerate(chunks):

            record = {
                "chunk_id": f"rail_{index:06d}",
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "page": chunk.metadata.get(
                        "page",
                        0
                    ) + 1,
                },
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Saved chunks to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()