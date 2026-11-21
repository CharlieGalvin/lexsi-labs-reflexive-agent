# EMNLP 2023 (*ACL) Paper Writing Guidelines

## General Setup
* **Document Class:** `\documentclass[11pt]{article}`, with `\pdfoutput=1` in the first lines. Compile with pdfLaTeX.
* **Style:** `\usepackage[review]{emnlp2023}` (style file `emnlp2023.sty`, bibliography style `acl_natbib.bst`). Also loaded: times, latexsym, T1 fontenc, utf8 inputenc, microtype, inconsolata, graphicx, booktabs, amsmath. Do not add packages that change fonts, margins or spacing.
* **Engine:** pdfLaTeX.

## Anonymity
* Review version is double-blind. Use the placeholder author "Ambitious AI Researcher, AI Research Institute, researcher@institute.ai". Refer to own prior work in the third person. Do not include repository URLs that reveal identity.

## Paper Length
* **Long paper:** at most **8 pages** of main text (abstract through conclusion).
* **Limitations** (mandatory, `\section*{Limitations}`, after the conclusion, before references) and **Ethics Statement** do not count toward the limit. Papers without a Limitations section are desk-rejected.
* References and appendices are unlimited and do not count.

## Layout
* Two-column A4 layout set by the style file. Do not change margins, column width, font sizes, or caption sizes.
* Abstract: one paragraph, at most 200 words.
* Use `figure`/`table` for single-column floats and `figure*`/`table*` for full-width floats.
* Table captions go above or below consistently; do not override default caption sizes.

## Citations
* natbib: `\citet{key}` → "Author (Year)"; `\citep{key}` → "(Author, Year)"; `\citealp{key}` → "Author, Year".
* Bibliography:
  ```latex
  \bibliography{references}
  \bibliographystyle{acl_natbib}
  ```
* BibTeX entries must not contain Unicode; include `doi` or `url` fields where possible. Only cite works that exist; every citation must be a real, verifiable paper.

## Figures
* All figures live in the `figures` directory. Always use the `figures/` prefix in `\includegraphics` (e.g. `\includegraphics[width=\columnwidth]{figures/heal-loop.png}`).

## Numbers
* Every number in the paper must come from the experimental log exactly as stated there (same rounding, same denominators, same confidence intervals). Do not invent, round differently, or extrapolate numbers.

## Required end matter
1. `\section*{Limitations}` (mandatory).
2. `\section*{Ethics Statement}`: must declare that the paper was drafted with AI assistance and checked by the authors.

## Appendix
* Use `\appendix` after the bibliography.
