# 1. Use a lightweight Python base image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Enable uv's optimization flags (compiles bytecode to speed up container boot)
ENV UV_COMPILE_BYTECODE=1


# 4. Copy environment files first to leverage Docker's caching mechanism
COPY pyproject.toml uv.lock ./

# 5. Install production dependencies (skips dev/test packages for a smaller image)
RUN uv sync --frozen --no-dev --no-install-project

# 6. Copy the rest of your application code into the container
COPY main.py ./
COPY app/ ./app/
COPY data/ ./data/

# 7. Expose the port Litestar runs on
EXPOSE 8000

# 8. Set the default command to run your Litestar server production build
CMD ["uv", "run", "litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]