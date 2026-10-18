# Original paper and poster materials

- `paper.pdf`: the final author manuscript, copied unchanged from `NeSy_2026_overleaf/nesy2026.pdf`.
- `poster.pdf`: copied unchanged from `Poster_SymAlign-UAST.pdf`.
- `figures/framework_from_poster.png`: the original embedded framework diagram from poster page 1 (PDF image 0, object 7, 1686 × 375 pixels). It is extracted directly, without redrawing, cropping, relabelling, or generating new visual content. This is the README overview figure.
- `figures/poincare_ball_visualization.png`: the original paper's Poincaré visualization (Figure 1), copied unchanged from the final manuscript's figure assets.
- The remaining three PNG files are original dataset/ontology and entropy/logic figures from the final manuscript's figure assets.

To extract the poster's embedded images with Poppler:

```sh
mkdir -p /tmp/symalign-poster-images
pdfimages -f 1 -l 1 -png assets/poster.pdf /tmp/symalign-poster-images/figure
```

The README figure is `figure-000.png` (SHA-256: `56020130fb8ea47e1233864787ac2730386c39bfcbc635a26b7de82005e4efb3`). These original materials retain the authorship and applicable rights described in [DATA_NOTICE.md](../DATA_NOTICE.md).
