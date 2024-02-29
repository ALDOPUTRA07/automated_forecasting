# Makefile

##@ Testing
unit-tests:
	@pytest

unit-tests-cov:
	@coverage run -m pytest
	@coverage report -m

unit-tests-cov-report-html:
	@pytest --cov=arima_model --cov-report term-missing --cov-report=html

unit-tests-cov-fail:
	@pytest --cov=arima_model --cov-report term-missing --cov-report=html --cov-fail-under=75


##@ Formatting
format-black: ## black (code formatter)
	@black .

format-isort: ## isort (import formatter)
	@isort .

format: format-black format-isort ## run all formatters

##@ Linting
lint-black:
	@black . --check

lint-isort:
	@isort . --check

lint-flake8:
	@flake8 .

lint-mypy:
	@mypy ./arima_model

lint-mypy-report:
	@mypy ./arima_model --html-report ./mypy_html

lint: lint-black lint-isort lint-flake8 lint-mypy
	

##@ Clean-up
clean: ## remove output files from pytest & coverage
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf site
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	@find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	

##@ Releases
current-version: ## returns the current version
	@semantic-release version --current

next-version: ## returns the next version
	@semantic-release version --next

current-changelog: ## returns the current changelog
	@semantic-release changelog --released

next-changelog: ## returns the next changelog
	@semantic-release changelog --unreleased

publish-noop: ## publish command (no-operation mode)
	@semantic-release publish --noop