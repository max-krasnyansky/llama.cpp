import textwrap

with open("ggml/src/ggml-hexagon/htp/hmx-gated-delta-net-ops.c", "r") as f:
    code = f.read()

# Fix 1: Execute worker directly instead of queueing it.
old_queue = """
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_gdn_worker, &job));
                hmx_queue_pop(ctx->hmx_queue);

                hmx_queue_suspend(ctx->hmx_queue);
"""

new_queue = """
                // The queue manages its own HMX locking so we can push tasks and pop them cleanly
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_gdn_worker, &job));
                hmx_queue_pop(ctx->hmx_queue);
"""
# Since the reviewer said "Queue Mismanagement (Blocking): The AI pushes the HMX task to a queue, immediately calls hmx_queue_pop (which removes/discards the task rather than executing it), and then permanently suspends the queue with hmx_queue_suspend. Since the main thread already holds the HMX lock via HAP_compute_res_hmx_lock, it should simply invoke the worker function directly (i.e., hmx_gdn_worker(&job);)."

# Wait! hmx_queue_pop blocks and waits for completion in this implementation!
# But the reviewer said: "Since the main thread already holds the HMX lock via HAP_compute_res_hmx_lock, it should simply invoke the worker function directly (i.e., hmx_gdn_worker(&job);)."
# The reviewer also noted "Missing Output De-interleaving".

old_read_back = """
                // Read output back
                // vtcm_attn is interleaved. Since it's n_tiles x 1 tile (S_v x 32), we can just read the first column.
                for (uint32_t r = 0; r < S_v; ++r) {
                    size_t tile_idx = r / 32;
                    size_t row_in_tile = r % 32;
                    attn_data[t * S_v + r] = (float) vtcm_attn[tile_idx * HMX_FP16_TILE_N_ELMS + row_in_tile];
                }
"""

new_read_back = """
                // Extract row 0 from HMX column-major tiles
                // In HMX output, each 32x32 tile is essentially column-major, where each column is 32 contiguous elements.
                for (uint32_t c = 0; c < S_v; ++c) {
                    size_t tile_idx = c / 32;
                    size_t col_in_tile = c % 32;
                    attn_data[t * S_v + c] = (float) vtcm_attn[tile_idx * HMX_FP16_TILE_N_ELMS + col_in_tile * 32 + 0];
                }
"""

new_queue_direct = """
                HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
                hmx_gdn_worker(&job);
                HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
"""

# Apply fixes
code = code.replace(old_read_back, new_read_back)
code = code.replace(old_queue, new_queue_direct)
# remove the commented out locks:
code = code.replace("// HAP_compute_res_hmx_lock(ctx->vtcm_rctx); // handled by queue", "")
code = code.replace("// HAP_compute_res_hmx_unlock(ctx->vtcm_rctx); // handled by queue", "")

with open("ggml/src/ggml-hexagon/htp/hmx-gated-delta-net-ops.c", "w") as f:
    f.write(code)
