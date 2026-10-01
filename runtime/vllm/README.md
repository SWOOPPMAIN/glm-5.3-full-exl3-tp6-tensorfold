# vLLM integration source

Source snapshot from Swoopp's P24-era full GLM TP6 work. The production image
is identified in [the recipe](../../recipes/vllm-tp6/README.md); these files
alone are not a complete image build context.

- `node.sh`: mounted serving entrypoint; validates the rank/checkpoint contract.
- `amos_*.py`: TP6 integration and optimization modules, including diagnostic helpers.
- `patch_*.py`, `install_*.py`: source-pinned installers for individual experiments.
- `memguard.py`: exact-container host memory/liveness protection.
- `rocenante/`: imported b12x/vLLM communication sources, original notices and pins.
- `e3/`: original Mia → Kindling grouped expert source with Swoopp native
  activation and ordered-reduction changes. AGPL-3.0; vendor headers retain MIT.

Several diagnostic modules and the early shared-expert experiment are inactive
in P24. Use the recorded image and tuning to identify the serving configuration.
Do not run all installers in filename order or apply them to arbitrary vLLM versions.
