        # MIT © 2025 MSc Candidate
        FROM python:3.10-slim
        ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
        WORKDIR /app

        # sistema deps para NetCDF & CDO
        RUN apt-get update && apt-get install -y --no-install-recommends \
                libproj-dev proj-data proj-bin cdo && \
            rm -rf /var/lib/apt/lists/*

        COPY pyproject.toml ./
        RUN pip install --upgrade pip && pip install -r <(python - <<'EOF'
import tomllib, pathlib, sys
deps = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']
print('\n'.join(deps))
EOF
        )

        COPY src ./src
        COPY configs ./configs
        ENTRYPOINT ["python", "-m", "src.models.train_xgb", "--config", "configs/default.yaml"]
