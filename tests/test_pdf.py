import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.errors import AppError
from app.ingest import extract_chunks


def make_pdf(texts, password=None):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_page_references(client):
    raw = make_pdf(["Refund within 30 days.", "Shipping takes 5 days."])
    response = client.post("/api/documents", files={"file": ("policy.pdf", raw, "application/pdf")})
    assert response.status_code == 201
    answer = client.post("/api/ask", json={"question": "Shipping?", "top_k": 1}).json()
    assert answer["sources"][0]["page"] == 2
    assert "Shipping" in answer["sources"][0]["text"]


def test_encrypted_pdf():
    with pytest.raises(AppError, match="Password-protected"):
        extract_chunks("locked.pdf", make_pdf(["Secret"], "password"), 1000)


def test_blank_pdf_needs_ocr():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(AppError, match="OCR"):
        extract_chunks("scan.pdf", buffer.getvalue(), 1000)
