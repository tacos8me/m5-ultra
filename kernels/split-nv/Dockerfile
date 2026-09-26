# RTX-side engine image for the DeepSeek-V4.1-Flash Mac + RTX split, from public parts only.
# Base: LMSYS DeepSeek-V4.1 dev image (torch 2.13.0+cu130, flashinfer 0.6.18), pinned by digest.
FROM lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
COPY sglang-dsv41-split.patch /tmp/
RUN cd /sgl-workspace/sglang/python \
 && git apply --whitespace=nowarn /tmp/sglang-dsv41-split.patch \
 && rm /tmp/sglang-dsv41-split.patch
