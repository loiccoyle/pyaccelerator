PKGNAME=accelerator

test:
	pytest tests

test-cov:
	pytest --cov=$(PKGNAME) tests --cov-report term-missing

format:
	isort . && black .

.PHONY: test test-cov format
