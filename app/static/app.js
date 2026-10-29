const $ = (selector) => document.querySelector(selector);
const selected = new Set();
let uploading = false;
let asking = false;
let documents = [];

function status(selector, message, error = false) {
  const element = $(selector);
  element.textContent = message;
  element.classList.toggle('error', error);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 204) return null;
  let data;
  try { data = await response.json(); }
  catch { throw new Error('The server returned an unreadable response. Check the terminal.'); }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'Please check your input and retry.';
    throw new Error(detail);
  }
  return data;
}

function updateScope() {
  $('#scope-label').textContent = selected.size ? `Searching ${selected.size} selected document(s)` : 'Searching your whole library';
}

function renderDocuments() {
  const container = $('#documents');
  container.replaceChildren();
  $('#document-count').textContent = String(documents.length);
  if (!documents.length) {
    const message = document.createElement('p');
    message.className = 'empty-library';
    message.textContent = 'Your next insight starts with your first document.';
    container.append(message);
  }
  documents.forEach((doc) => {
    const fragment = $('#document-template').content.cloneNode(true);
    fragment.querySelector('strong').textContent = doc.name;
    fragment.querySelector('strong').title = doc.name;
    fragment.querySelector('small').textContent = `${doc.chunks} passages · ${(doc.size / 1024).toFixed(1)} KB`;
    const checkbox = fragment.querySelector('input');
    checkbox.checked = selected.has(doc.id);
    checkbox.disabled = asking;
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) selected.add(doc.id); else selected.delete(doc.id);
      updateScope();
    });
    const button = fragment.querySelector('.delete');
    button.disabled = asking;
    button.setAttribute('aria-label', `Delete ${doc.name}`);
    button.addEventListener('click', async () => {
      if (!confirm(`Delete "${doc.name}" from your local library?`)) return;
      button.disabled = true;
      try {
        await api(`/api/documents/${doc.id}`, {method: 'DELETE'});
        selected.delete(doc.id);
        await refresh();
        status('#upload-status', 'Document deleted. Earlier answers remain on this page.');
      } catch (error) { status('#upload-status', error.message, true); button.disabled = false; }
    });
    container.append(fragment);
  });
  updateScope();
}

async function refresh() {
  const data = await api('/api/documents');
  documents = data.documents;
  for (const id of selected) if (!documents.some((doc) => doc.id === id)) selected.delete(id);
  renderDocuments();
}

async function upload(file) {
  if (!file || uploading) return;
  if (file.size > 20 * 1024 * 1024) { status('#upload-status', 'Choose a file under 20 MB.', true); return; }
  uploading = true;
  $('#file-input').disabled = true;
  status('#upload-status', 'Indexing your document… First use downloads the local embedding model and can take a few minutes.');
  const body = new FormData();
  body.append('file', file);
  try {
    const data = await api('/api/documents', {method: 'POST', body});
    await refresh();
    status('#upload-status', data.duplicate ? 'This document is already in your library.' : `Ready. ${data.document.chunks} passages indexed.`);
  } catch (error) { status('#upload-status', error.message, true); }
  finally { uploading = false; $('#file-input').disabled = false; $('#file-input').value = ''; }
}

$('#file-input').addEventListener('change', (event) => upload(event.target.files[0]));
const dropzone = $('#dropzone');
['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', (event) => upload(event.dataTransfer.files[0]));
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => {
  $('#question').value = button.dataset.question;
  $('#question').focus();
}));
$('#question').addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') $('#question-form').requestSubmit();
});

function addAnswer(question, data) {
  $('#welcome').hidden = true;
  const turn = document.createElement('article');
  turn.className = 'turn';
  const bubble = document.createElement('div');
  bubble.className = 'question-bubble';
  bubble.textContent = question;
  const label = document.createElement('p');
  label.className = 'answer-label';
  label.textContent = `FOLIO / ${data.model}`;
  const answer = document.createElement('div');
  answer.className = 'answer-text';
  // Use textContent for all document and model content; never inject HTML.
  answer.textContent = data.answer;
  turn.append(bubble, label, answer);
  const note = document.createElement('p');
  note.className = 'source-note';
  note.textContent = data.citation_warning ? 'No inline citations returned. These excerpts were retrieved; verify the answer against them.' : 'Retrieved evidence · Expand a passage to check the answer.';
  turn.append(note);
  const sources = document.createElement('div');
  sources.className = 'source-list';
  data.sources.forEach((source) => {
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `[${source.label}] ${source.filename}${source.page ? ` · page ${source.page}` : ''} · ${source.cited ? 'cited' : 'retrieved'}`;
    const text = document.createElement('p');
    text.textContent = source.text;
    details.append(summary, text);
    sources.append(details);
  });
  turn.append(sources);
  $('#conversation').append(turn);
  turn.scrollIntoView({behavior: 'smooth', block: 'start'});
}

$('#question-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = $('#question').value.trim();
  if (asking || !question) return;
  if (!documents.length) { status('#ask-status', 'Upload a document first.', true); return; }
  asking = true;
  $('#ask-button').disabled = true;
  $('#question').disabled = true;
  renderDocuments();
  status('#ask-status', 'Finding relevant passages and preparing an answer… This may take up to 3 minutes.');
  try {
    const data = await api('/api/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question, document_ids: [...selected]})});
    addAnswer(question, data);
    $('#question').value = '';
    status('#ask-status', '');
  } catch (error) { status('#ask-status', error.message, true); }
  finally {
    asking = false;
    $('#ask-button').disabled = false;
    $('#question').disabled = false;
    renderDocuments();
    $('#question').focus();
  }
});

async function initialize() {
  try {
    const health = await api('/api/health');
    const local = health.provider === 'ollama';
    $('#model-status').textContent = local ? `Free local AI · ${health.model}` : (health.grok_configured ? `Grok configured · ${health.model}` : 'API key needed');
    $('#config-banner').hidden = local || health.grok_configured;
    await refresh();
  } catch (error) { status('#ask-status', `Unable to connect. ${error.message}`, true); }
}
initialize();
