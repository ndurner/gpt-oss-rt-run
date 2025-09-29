SHELL := /bin/sh

ARCHIVE := experiments-raw.tar.xz

.PHONY: experiments-dist

experiments-dist:
	python3 scripts/create_experiments_archive.py $(ARCHIVE)
