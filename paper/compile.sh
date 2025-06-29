#!/bin/bash

# Compile the NOVA experiments paper
echo "Compiling NOVA experiments paper..."

# Run pdflatex twice to resolve references
pdflatex nova_experiments.tex
pdflatex nova_experiments.tex

# Clean up auxiliary files (optional)
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk

echo "Compilation complete! Check nova_experiments.pdf" 