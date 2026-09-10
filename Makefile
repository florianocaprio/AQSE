.PHONY: build up down logs test prepare-demo demo acceptance soak clean

build:
	docker compose build

up: build
	docker compose up -d

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

test: build
	docker compose run --rm \
		-v ./docker-compose.yml:/workspace/docker-compose.yml:ro \
		-v ./Makefile:/workspace/Makefile:ro \
		backend python -m pytest
	docker compose run --rm backend ruff check .
	docker compose run --rm --no-deps frontend sh -c "CI=true pnpm install --frozen-lockfile && pnpm run typecheck && pnpm run lint && pnpm run test && pnpm run build"

prepare-demo: build
	docker compose run --rm --no-deps backend python -m scripts.prepare_demo

# Starts saved artifacts only; preparation, training, and TEST access remain explicit.
demo: build
	docker compose up -d --wait --wait-timeout $${AQSE_DEMO_WAIT_SECONDS:-120}

acceptance:
	docker compose exec -T backend python -m scripts.demo_acceptance \
		--base-url http://127.0.0.1:8000 \
		--frontend-url http://frontend:3000 \
		--timeout-s $${AQSE_ACCEPTANCE_TIMEOUT_SECONDS:-90} \
		--report-dir /artifacts/validation

# Ten real wall-clock minutes by default against the running observation worker.
soak:
	docker compose exec -T backend python -m scripts.demo_soak \
		--base-url http://127.0.0.1:8000 \
		--duration-s $${SOAK_DURATION_SECONDS:-600} \
		--nodes $${SOAK_NODE_COUNT:-8} \
		--seed $${SOAK_SEED:-8600} \
		--sample-interval-s $${SOAK_SAMPLE_INTERVAL_SECONDS:-5} \
		--report-dir /artifacts/validation

clean:
	docker compose down --remove-orphans
	rm -rf .pnpm-store frontend/.pnpm-store frontend/node_modules frontend/dist frontend/coverage backend/.pytest_cache backend/.ruff_cache backend/app/__pycache__ backend/tests/__pycache__
	find backend -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f frontend/*.tsbuildinfo
