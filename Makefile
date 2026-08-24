# BT-DKGRec-GCN — Makefile
# Cac buoc chua trien khai bao loi ro rang thay vi im lang chay qua.

PY        := .venv/bin/python
PIP       := .venv/bin/pip
COHORT    ?= original
MODEL     ?= bt_dkgrec
SEED      ?= 2020
DEVICE    ?= cuda

.PHONY: help setup preprocess graph train evaluate multiseed tables neo4j app test clean-venv

help:
	@echo "BT-DKGRec-GCN"
	@echo ""
	@echo "  Moi thuc nghiem chay tren Colab: notebooks/run_all.ipynb"
	@echo "  Tren VPS chi dung: setup, test, tables, neo4j, app"
	@echo ""
	@echo "  make setup                                    # venv + thu vien loi + kiem tra Buoc 1"
	@echo "  make test                                     # pytest"
	@echo "  make preprocess COHORT=original               # [Buoc 2]"
	@echo "  make graph COHORT=original                    # dung ca 3 bien the"
	@echo "  make graph-one COHORT=original MODEL=bt_dkgrec"
	@echo "  make train MODEL=bt_dkgrec SEED=2020 COHORT=original   # [Buoc 6]"
	@echo "  make evaluate RUN=<run_id>                    # [Buoc 5]"
	@echo "  make multiseed                                # [Buoc 9]"
	@echo "  make tables                                   # [Buoc 9]"
	@echo "  make neo4j / make app                         # [Buoc 10]"
	@echo ""
	@echo "  Bien: COHORT=$(COHORT) MODEL=$(MODEL) SEED=$(SEED) DEVICE=$(DEVICE)"

# ── Buoc 1 ────────────────────────────────────────────────────────────
$(PY):
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip

setup: $(PY)
	$(PIP) install --quiet -r requirements.txt
	$(PY) scripts/00_check_setup.py

test: $(PY)
	$(PY) -m pytest

clean-venv:
	rm -rf .venv

# ── Cac buoc sau ──────────────────────────────────────────────────────
preprocess:
	@test -f scripts/01_preprocess.py || { echo "CHUA TRIEN KHAI: scripts/01_preprocess.py (Buoc 2)"; exit 1; }
	$(PY) scripts/01_preprocess.py --cohort $(COHORT)

# Dung ca ba bien the co graph (lightgcn, static_kg_gcn, bt_dkgrec) de bao dam
# chung duoc sinh tu cung mot tap interim trong cung mot lan chay.
graph:
	$(PY) scripts/02_build_graph.py --cohort $(COHORT) --all

graph-one:
	$(PY) scripts/02_build_graph.py --cohort $(COHORT) --model $(MODEL)

train:
	$(PY) scripts/03_train.py --model $(MODEL) --cohort $(COHORT) --seed $(SEED) --device $(DEVICE)

evaluate:
	@test -f scripts/04_evaluate.py || { echo "CHUA TRIEN KHAI: scripts/04_evaluate.py (Buoc 5)"; exit 1; }
	@test -n "$(RUN)" || { echo "Thieu tham so: make evaluate RUN=<run_id>"; exit 1; }
	$(PY) scripts/04_evaluate.py --run $(RUN)

multiseed:
	@test -f scripts/05_run_multiseed.py || { echo "CHUA TRIEN KHAI: scripts/05_run_multiseed.py (Buoc 9)"; exit 1; }
	$(PY) scripts/05_run_multiseed.py

tables:
	@test -f scripts/06_make_tables.py || { echo "CHUA TRIEN KHAI: scripts/06_make_tables.py (Buoc 9)"; exit 1; }
	$(PY) scripts/06_make_tables.py

neo4j:
	@test -f scripts/07_export_neo4j.py || { echo "CHUA TRIEN KHAI: scripts/07_export_neo4j.py (Buoc 10)"; exit 1; }
	$(PY) scripts/07_export_neo4j.py --cohort $(COHORT)

app:
	@test -f app/main.py || { echo "CHUA TRIEN KHAI: app/main.py (Buoc 10)"; exit 1; }
	.venv/bin/streamlit run app/main.py
