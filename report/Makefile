# Makefile do kompilacji raportu LaTeX

# Nazwa głównego pliku (bez rozszerzenia)
MAIN = raport

# Kompilator LaTeX
LATEX = pdflatex

# Opcje kompilacji
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Domyślny target
all: $(MAIN).pdf

# Kompilacja PDF (uruchom dwukrotnie dla TOC i referencji)
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Czyszczenie plików tymczasowych
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg

# Czyszczenie wszystkiego (włącznie z PDF)
cleanall: clean
	rm -f $(MAIN).pdf

# Otwarcie PDF
open: $(MAIN).pdf
	open $(MAIN).pdf || xdg-open $(MAIN).pdf || evince $(MAIN).pdf

# Pomoc
help:
	@echo "Dostępne komendy:"
	@echo "  make          - Kompiluje raport.pdf"
	@echo "  make clean    - Usuwa pliki tymczasowe"
	@echo "  make cleanall - Usuwa wszystko (łącznie z PDF)"
	@echo "  make open     - Otwiera raport.pdf"

.PHONY: all clean cleanall open help
