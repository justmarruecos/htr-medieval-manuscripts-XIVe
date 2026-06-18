#!/bin/bash
# Compiler l'article en PDF
# Nécessite pdflatex (inclus dans MacTeX ou TeX Live)
# Installation: brew install --cask mactex-no-gui

cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode article_htr.tex
pdflatex -interaction=nonstopmode article_htr.tex  # 2ème passe pour les refs
echo ""
echo "✅ PDF généré: article_htr.pdf"
