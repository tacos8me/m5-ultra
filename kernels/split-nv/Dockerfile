# RTX-side engine image for the DeepSeek-V4.1-Flash Mac + RTX split, from public parts only.
# Base: LMSYS DeepSeek-V4.1 dev image (torch 2.13.0+cu130, flashinfer 0.6.18), pinned by digest.
# The build fails unless the patched SGLang tree matches the tested tree file for file (TREE.sha256).
FROM lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
COPY sglang-dsv41-split.patch flashinfer-sparse-mla-sm120.patch TREE.sha256 /tmp/
RUN cd /sgl-workspace/sglang/python \
 && patch -p1 -E --no-backup-if-mismatch < /tmp/sglang-dsv41-split.patch \
 && sha256sum --quiet -c /tmp/TREE.sha256
# FlashInfer SM120 sparse-MLA prefill: dispatch extra_page_block_size 128/256 (V4.1's compressed layers).
# The AOT .so shadows the sources, so rebuild it from the patched sources (sm_120f, -O3, no GPU needed).
ARG FI=/opt/sglang/lib/python3.12/site-packages/flashinfer
ARG AOT=/opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120
RUN cd $FI && patch -p1 --no-backup-if-mismatch < /tmp/flashinfer-sparse-mla-sm120.patch \
 && rm -f $AOT/sparse_mla_sm120.so \
 && FLASHINFER_CUDA_ARCH_LIST=12.0f python3 -c "from flashinfer.jit.mla import gen_sparse_mla_sm120_module as g; s=g(); s.build(); print(s.jit_library_path)" > /tmp/so \
 && cp "$(tail -1 /tmp/so)" $AOT/sparse_mla_sm120.so \
 && rm -rf /root/.cache/flashinfer /tmp/so /tmp/*.patch /tmp/TREE.sha256
