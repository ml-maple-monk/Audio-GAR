# AudioGAR project page

Static project page for **AudioGAR: Bridging Reconstruction and Generation in Latent Audio Generative Models**. Open `index.html` directly, or serve this directory with any static HTTP server. There is no build step or third-party runtime dependency.

From the repository root:

```sh
python -m http.server 8000 --directory docs
```

Then visit `http://localhost:8000`. For Python versions before 3.7, run `python -m http.server 8000` from inside `docs`.

For GitHub Pages, select the desired branch and `/docs` as the publishing folder in the repository's Pages settings.

## Content and assets

- `index.html`: paper narrative, the first four main-text figures, the full 29-row reconstruction-generation table, main results, ablation, and citation.
- `styles.css`: layout and typography based on `../docs2`, with responsive tables, figures, and keyboard focus styles.
- `script.js`: optional dataset filtering and citation copying; all research content remains available without JavaScript.
- `assets/AudioGAR.pdf`: supplied paper, the source of the page's narrative and results.
- `assets/AudioGAR.bib`: citation matching the visible BibTeX block. Uses `@article` with `journal = {Arxiv}`, following the authors' requested citation format.

The author-provided `lda-MAF-main.pdf`, `AudioGAR_pipeline.pdf`, and `latent-fd-AudioCaps.pdf` have matching PNG exports rendered at 300 dpi. The remaining PNGs are 300 dpi crops from `AudioGAR.pdf`: Figure 2 on page 5, Figure 4(b) on page 6, Figures 5 and 6 on page 9, and Figure 7 on page 10. The page displays Figures 1 through 4 in paper order; exports for Figures 5 through 7 remain in assets but are not displayed. The original PDFs remain available through the figure links and paper link.

The gap table transcribes the supplied LaTeX table, including checkpoint links. The page's main-results table shows the reproduced AudioX and TangoMusic baselines and their AudioGAR variants from paper Table 3. The cost table comes from paper Table 1; the ablation table shows the FD/FAD columns from paper Table 4.

The supplied `paper-blog-post` writing guidance governs the narrative. The requested static `docs2` structure replaces its Jekyll-specific file layout. No audio examples were supplied, so the page makes no listening-demo claims.
