# Upload to Overleaf

1. Create a new project > Upload Project and select the ZIP beside this folder.
2. Set main.tex as the main document and use pdfLaTeX.
3. Recompile and download the PDF. The default document is print-friendly and uses three side-by-side cards with real still frames.

Printing cannot preserve motion. Numbered PNG frames are included for all three GIFs; no online splitter is needed. To add optional interactive animation, add \usepackage{animate} to the preamble and insert one of the commands below where desired. PDF animation requires a compatible viewer, such as Adobe Acrobat Reader desktop; Overleaf/browser previews may show only a still. Keep the still-image cards for printing. Animation is optional and is not enabled in main.tex.

\animategraphics[controls,loop,width=0.85\linewidth]{10}{frames/stage1-}{0}{30}

\animategraphics[controls,loop,width=0.85\linewidth]{10}{frames/stage2-}{0}{60}

\animategraphics[controls,loop,width=0.85\linewidth]{10}{frames/stage3-}{0}{60}

## Editing

Edit main.tex for the Overleaf document. DOCUMENTATION.md in the repository root is the GitHub-friendly version. Both describe measured results, distinguish the original telemetry model from teacher imitation, and identify unfinished rubric items. The original F1_Racing_Simulator.pdf was used as planning context; it was not overwritten.

## Evidence

Historical results come from the existing local analysis pack and benchmark outputs. The completed follow-up study in output/completed_study supplies training-only tuning, the explicit ensemble comparison and the fresh Spa evaluation. The Git remote supplied the repository URL; this operation does not push or publish changes.


## Concise presentation structure

main.tex is the concise, print-friendly version organized around eight slide-sized sections. detailed.tex retains the longer supporting document. Set the desired file as the Overleaf main document. PRESENTATION_PLAN.md gives timing, visuals and speaker cues; PROJECT_EVIDENCE.md holds full supporting detail. Rebuild this version with python build_concise_documentation.py from the repository root.
