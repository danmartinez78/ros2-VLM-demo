# Derived from the known-good Thor FlashRT image built for Cosmos3 Reasoner.
# Build context must contain a pinned NVIDIA cosmos-framework checkout.
FROM flashrt:cosmos3-thor

ARG COSMOS_FRAMEWORK_DIR=cosmos-framework
COPY ${COSMOS_FRAMEWORK_DIR} /opt/cosmos-framework

# cosmos-framework 2026-07-20 imports boto3 transitively from its processor
# package even for local checkpoints. qwen-vl-utils is retained for parity with
# the validated video preprocessing setup.
RUN python -m pip install --no-cache-dir /opt/cosmos-framework \
    && python -m pip install --no-cache-dir qwen-vl-utils boto3

ENV PYTHONPATH=/opt/cosmos-framework
