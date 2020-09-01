PKGNAME=accelerator

test:
	pytest tests

test-cov:
	pytest --cov=$(PKGNAME) tests --cov-report term-missing

.PHONY: test test-cov
