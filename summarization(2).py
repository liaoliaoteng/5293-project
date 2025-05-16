# summarization_cli.py

import subprocess
import argparse
from typing import List
import pdfplumber
from docx import Document
import textwrap

    # —— 1. Text extraction —— #
def extract_text(file_path: str) -> str:
    """
    Automatically call the corresponding extraction function according to the file suffix.
    """
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    if file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    # Default is plain text
    with open(file_path, encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)

def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# —— 2. Text segmentation —— #
def chunk_text(text: str, max_chars: int = 20000) -> List[str]:
    """
    Split long text into characters to avoid the prompt being too long.
    """
    # Here we simply split by paragraphs and then pack by length
    paras = text.split("\n")
    chunks: List[str] = []
    current = []
    count = 0
    for p in paras:
        l = len(p)
        if count + l > max_chars and current:
            chunks.append("\n".join(current))
            current = [p]
            count = l
        else:
            current.append(p)
            count += l
    if current:
        chunks.append("\n".join(current))
    return chunks

# —— 3. Call Ollama CLI to complete the summary —— #
def summarize_with_ollama(text: str, model: str) -> str:
    prompt = (
        "The following is the summary content：\n"
        f"{text}\n\n"
        "Please extract the key points of this passage in English and output a concise summary："
    )
    # Call Ollama CLI and specify decoding parameters
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="ignore",
        check=False
    )

    # Prioritize stdout; if empty, use stderr
    output = proc.stdout.strip() if proc.stdout else proc.stderr.strip()
    return output

def summarize(text: str, model: str) -> str:
    chunks = chunk_text(text)
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        print(f">>> Summarizing chunk {idx}/{len(chunks)} ...")
        s = summarize_with_ollama(chunk, model)
        # If the output of Ollama CLI still contains prompt, you can split it here:
        # s = s.split("Output a concise summary:")[-1].strip()
        summaries.append(s)
    return "\n\n".join(summaries)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Pending summary documents (.pdf/.docx/.txt)")
    parser.add_argument("--model", default="deepseek-r1:14b")
    args = parser.parse_args()

    raw = extract_text(args.file)
    print("=== The first 500 words of the original text ===\n", raw[:500], "\n")
    summary = summarize(raw, args.model)
    print("=== Final Summary ===\n", summary)

if __name__ == "__main__":
    main()