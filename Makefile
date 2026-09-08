.PHONY: build up down logs test clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

test:
	docker compose run --rm backend python -m pytest
	docker compose run --rm frontend pnpm run build

clean:
	docker compose down --remove-orphans
	rm -rf frontend/dist backend/.pytest_cache backend/app/__pycache__ backend/tests/__pycache__
