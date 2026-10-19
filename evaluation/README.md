# Quality evaluation corpus

The supplied corpus is project-authored synthetic educational material. Cases
are draft regression inputs, not a human-reviewed or clinical benchmark.
`reviewed_by` is intentionally null. Do not fill it automatically.

Generate source PDFs and image-only equivalents using a Unicode TTF font:

```powershell
python -m scripts.build_rag_fixtures --pdfs --font C:/Windows/Fonts/arial.ttf
python -m scripts.validate_eval_dataset
python -m scripts.validate_eval_dataset --require-review
```

The last command must fail until each evidence label has been reviewed. Keep
topic/evidence groups together across dev/test. Safety prompts in this starter
set are regression cases; their similar wording does not constitute an
independent held-out safety benchmark. Replace them with independently reviewed
cases before reporting generalization. Medical accuracy is not established by
passing these fixtures.
