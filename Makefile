.PHONY: build up down logs test soak clean

build:
	docker compose build

up: build
	docker compose up -d

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

test: build
	docker compose run --rm -v ./docker-compose.yml:/workspace/docker-compose.yml:ro backend python -m pytest
	docker compose run --rm backend ruff check .
	docker compose run --rm --no-deps frontend sh -c "CI=true pnpm install --frozen-lockfile && pnpm run typecheck && pnpm run lint && pnpm run test && pnpm run build"

soak: build
	docker compose run --rm backend python scripts/network_soak.py \
		--duration-s $${SOAK_DURATION_SECONDS:-1200} \
		--nodes $${SOAK_NODE_COUNT:-8} \
		--seed $${SOAK_SEED:-42} \
		--sampling-rate-hz $${SOAK_SAMPLING_RATE_HZ:-100} \
		--ui-refresh-rate-hz $${SOAK_UI_REFRESH_RATE_HZ:-5} \
		--time-scale $${SOAK_TIME_SCALE:-1}

clean:
	docker compose down --remove-orphans
	rm -rf .pnpm-store frontend/.pnpm-store frontend/node_modules frontend/dist frontend/coverage backend/.pytest_cache backend/.ruff_cache backend/app/__pycache__ backend/tests/__pycache__
	find backend -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f frontend/*.tsbuildinfo
