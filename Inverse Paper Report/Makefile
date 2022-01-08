
default: report.pdf clean

report.pdf: report.tex notation.tex tex/* figures/* refs.bib
	pdflatex report.tex
	pdflatex report.tex
	bibtex report
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f *.{aux,out,log,bbl,blg,vtc}
