"""Count-based variants of MMH3 Tiled Diffusion and MMH3 Temporal Split Params.

Instead of a tile / chunk size, the user picks how many pieces to cut and the
overlap as a fraction of a piece; sizes are solved from the actual latent.
MMH3 Auto Split Planner picks both counts from a per-forward token budget.
"""

from comfy_api.latest import io

from .h3_latent_upscaler import _compute_upscale_target
from .nodes import H3_TEMPORAL_PARAM, H3_UPSCALE_PARAM, audio_range, compute_segments, frames_for_tokens
from .tiled_diffusion import _H3TiledDiffusionImpl, _ceil2, _tile_starts

# n_tiles -> (tiles along the long edge, tiles along the short edge)
TILE_GRIDS = {"1": (1, 1), "2": (2, 1), "3": (3, 1), "4": (2, 2), "6": (3, 2)}


def _tile_size(total, n, overlap):
    """Smallest even tile size so `n` tiles overlapping by `overlap` cover `total`."""
    return _ceil2(-(-(total + (n - 1) * overlap) // n))


def _solve_tiles(H, W, n_tiles, overlap_frac):
    """(tile_h, tile_w, overlap) in latent units for `n_tiles` tiles on an H x W latent."""
    long_n, short_n = TILE_GRIDS[n_tiles]
    rows, cols = (short_n, long_n) if W >= H else (long_n, short_n)
    # one overlap shared by both axes (base _build contract): grow it until
    # every split axis overlaps by at least overlap_frac of its tile
    ov = 0
    while True:
        th, tw = _tile_size(H, rows, ov), _tile_size(W, cols, ov)
        if all(ov >= overlap_frac * t for t, n in ((th, rows), (tw, cols)) if n > 1):
            return th, tw, ov
        ov += 2


def _solve_temporal(tv, chunk_count, overlap_frac):
    """(bounds, chunk_length, temporal_overlap) cutting `tv` video tokens into
    at most `chunk_count` chunks on the 17-frame grid."""
    frame_count = frames_for_tokens(tv)
    n = chunk_count
    # grow the overlap on the 17-frame grid until it reaches overlap_frac
    # of the chunk; per overlap, grow the chunk until the real split
    # (boundaries snap to keyframe tokens) fits in chunk_count chunks
    overlap = 0
    while True:
        chunk_length = max(overlap + 17, -(-(frame_count + (n - 1) * overlap) // (17 * n)) * 17)
        bounds, _ = compute_segments(tv, chunk_length, overlap)
        while len(bounds) > n:
            chunk_length += 17
            bounds, _ = compute_segments(tv, chunk_length, overlap)
        if overlap >= overlap_frac * chunk_length:
            return bounds, chunk_length, overlap
        overlap += 17


class _AutoTiledDiffusionImpl(_H3TiledDiffusionImpl):
    """Solves tile_h / tile_w / overlap from the latent size, then builds the
    tiles exactly like the base implementation."""

    def __init__(self, method, n_tiles, overlap_frac):
        super().__init__(method, 0, 0, 0)
        self.n_tiles = n_tiles
        self.overlap_frac = overlap_frac

    def _build(self, payload, video, audio, c):
        H, W = video.shape[3], video.shape[4]
        self.tile_h, self.tile_w, self.overlap = _solve_tiles(H, W, self.n_tiles, self.overlap_frac)
        return super()._build(payload, video, audio, c)


class MMH3TiledDiffusionAuto(io.ComfyNode):
    """MMH3 Tiled Diffusion with a tile count instead of a tile size."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TiledDiffusionAuto",
            display_name="MMH3 Tiled Diffusion Auto (Experimental)",
            category="model/latent/minimax",
            description=(
                "Same as 'MMH3 Tiled Diffusion (Experimental)', but the tile grid "
                "is chosen by tile count and the tile size is solved from the "
                "latent. The long edge follows the frame orientation, so portrait "
                "video is handled automatically."
            ),
            search_aliases=["tiled diffusion auto", "h3 tiling auto"],
            inputs=[
                io.Model.Input("model",
                               tooltip="The MiniMax H3 diffusion model to patch."),
                io.Combo.Input("method", options=["gaussian", "uniform"], default="gaussian",
                               tooltip="Overlap blending weight per tile: gaussian (tiles dominate their center) or uniform (plain averaging in overlaps)."),
                io.Combo.Input("n_tiles", options=list(TILE_GRIDS), default="3",
                               tooltip="Number of tiles. 2 / 3: split the long edge. 4: 2x2. 6: 3 on the long edge x 2 on the short edge."),
                io.Float.Input("overlap_frac", default=0.5, min=0.0, max=0.5, step=0.05,
                               tooltip="Overlap between neighbouring tiles as a fraction of the tile size. Keep at least ~0.25 or the tiles decouple."),
            ],
            outputs=[
                io.Model.Output("model",
                                tooltip="The patched model; feed it into 'MMH3 Ultimate Upscale' (spatial_split_param unconnected) or any H3 sampler."),
            ],
        )

    @classmethod
    def execute(cls, model, method, n_tiles, overlap_frac) -> io.NodeOutput:
        patched = model.clone()
        patched.set_model_unet_function_wrapper(_AutoTiledDiffusionImpl(method, n_tiles, overlap_frac))
        return io.NodeOutput(patched)


class MMH3TemporalSplitParamsAuto(io.ComfyNode):
    """MMH3 Temporal Split Params with a chunk count instead of a chunk length."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalSplitParamsAuto",
            display_name="MMH3 Temporal Split Params Auto",
            category="model/latent/minimax",
            description=(
                "Same as 'MMH3 Temporal Split Params', but the latent is cut into "
                "chunk_count chunks; chunk_length and temporal_overlap are solved "
                "on the 17-frame grid from the latent's frame count."
            ),
            search_aliases=["h3 temporal params auto", "time split auto"],
            inputs=[
                io.Latent.Input("latent",
                                tooltip="The same MiniMax H3 AV latent fed into 'MMH3 Ultimate Upscale'."),
                io.Int.Input("chunk_count", default=4, min=1, max=100, step=1,
                             tooltip="Number of time chunks. Short videos may end up with fewer chunks, since boundaries snap to the 17-frame grid."),
                io.Float.Input("overlap_frac", default=0.33, min=0.0, max=0.5, step=0.01,
                               tooltip="Overlap between consecutive chunks as a fraction of the chunk length, rounded to a multiple of 17 frames."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="How much of the previous chunk's re-sampled boundary the frozen frame-0 anchor keeps: 1.0 = exact content, 0.999 = model default, 0.0 = no anchoring."),
            ],
            outputs=[
                H3_TEMPORAL_PARAM.Output("temporal_split_param",
                                         tooltip="Temporal split settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, latent, chunk_count, overlap_frac, anchor_strength) -> io.NodeOutput:
        tv = latent["samples"].tensors[0].shape[2]
        bounds, chunk_length, overlap = _solve_temporal(tv, chunk_count, overlap_frac)
        print(f"[MMH3 Temporal Split Params Auto] {len(bounds)} chunks over {frames_for_tokens(tv)} frames "
              f"(chunk_length={chunk_length}, temporal_overlap={overlap})")
        return io.NodeOutput({
            "chunk_length": chunk_length,
            "temporal_overlap": overlap,
            "anchor_strength": anchor_strength,
        })


def _max_tokens(bounds, H, W, tile_h, tile_w):
    """Largest per-forward token count over all chunks: one tile's video tokens
    per time token (plus the frame-0 anchor) and the chunk's audio tokens."""
    per_t = (_ceil2(min(tile_h, H)) // 2) * (_ceil2(min(tile_w, W)) // 2)
    return max((k1 - k0 + 1) * per_t + a1 - a0
               for k0, f0, k1, f1 in bounds
               for a0, a1 in [audio_range(f0, f1)])


class MMH3AutoSplitPlanner(io.ComfyNode):
    """Picks the fewest tiles, then the fewest time chunks, that keep every
    DiT forward under a token budget."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3AutoSplitPlanner",
            display_name="MMH3 Auto Split Planner",
            category="model/latent/minimax",
            description=(
                "Chooses the time chunk count and Tiled Diffusion tile count for "
                "'MMH3 Ultimate Upscale' from the latent length and sampling size. "
                "Time is split first; tiles are added only when chunks would get "
                "shorter than min_chunk_frames. With 1 tile or spatial_split off the "
                "model is passed through unpatched; with 1 chunk or temporal_split "
                "off temporal_split_param is empty, so the sampler takes its "
                "no-split path. Fails when no allowed split fits token_budget. "
                "Leave spatial_split_param unconnected."
            ),
            search_aliases=["auto split", "split planner", "h3 auto split"],
            inputs=[
                io.Model.Input("model",
                               tooltip="The MiniMax H3 diffusion model."),
                io.Latent.Input("latent",
                                tooltip="The same MiniMax H3 AV latent fed into 'MMH3 Ultimate Upscale'."),
                H3_UPSCALE_PARAM.Input("latent_upscale_param", optional=True,
                                       tooltip="The same latent_upscale_param fed into 'MMH3 Ultimate Upscale'. When connected, its width/height is the sampling size; otherwise the latent size is."),
                io.Int.Input("token_budget", default=44000, min=1000, max=1000000, step=1000,
                             tooltip="Max tokens per DiT forward (video tokens of one tile x chunk + audio). Calibrate from a run that fits in VRAM. The workflow stops when no allowed split fits."),
                io.Boolean.Input("spatial_split", default=True,
                                 tooltip="Allow Tiled Diffusion tiles. Off: the model is passed through unpatched."),
                io.Float.Input("tile_overlap_frac", default=0.5, min=0.0, max=0.5, step=0.05,
                               tooltip="Overlap between neighbouring tiles as a fraction of the tile size. Unused when spatial_split is off."),
                io.Combo.Input("method", options=["gaussian", "uniform"], default="gaussian",
                               tooltip="Tiled Diffusion overlap blending weight, used only when tiles > 1."),
                io.Boolean.Input("temporal_split", default=True,
                                 tooltip="Allow time chunks. Off: temporal_split_param is empty."),
                io.Int.Input("min_chunk_frames", default=119, min=17, max=100000, step=17,
                             tooltip="Shortest allowed time chunk in pixel frames (multiple of 17). Tiles are added instead of cutting time chunks shorter than this. Unused when temporal_split is off."),
                io.Float.Input("temporal_overlap_frac", default=0.25, min=0.0, max=0.5, step=0.01,
                               tooltip="Overlap between consecutive chunks as a fraction of the chunk length. Unused when temporal_split is off."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="Frame-0 anchor strength of each time chunk, used only when chunks > 1."),
            ],
            outputs=[
                io.Model.Output("model",
                                tooltip="Unchanged model for 1 tile, otherwise the Tiled Diffusion patched model."),
                H3_TEMPORAL_PARAM.Output("temporal_split_param",
                                         tooltip="Empty for 1 chunk, otherwise the temporal split settings for 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, model, latent, token_budget, spatial_split, tile_overlap_frac, method,
                temporal_split, min_chunk_frames, temporal_overlap_frac, anchor_strength,
                latent_upscale_param=None) -> io.NodeOutput:
        _, _, tv, H, W = latent["samples"].tensors[0].shape
        if latent_upscale_param is not None:
            H, W, _ = _compute_upscale_target(latent_upscale_param["width"], latent_upscale_param["height"], H, W)

        plan = None
        for n_tiles in (TILE_GRIDS if spatial_split else ["1"]):
            tile_h, tile_w, _ = _solve_tiles(H, W, n_tiles, tile_overlap_frac)
            for n in range(1, tv // 5 + 2 if temporal_split else 2):
                bounds, chunk_length, overlap = _solve_temporal(tv, n, temporal_overlap_frac)
                if len(bounds) > 1 and chunk_length < min_chunk_frames:
                    break
                tokens = _max_tokens(bounds, H, W, tile_h, tile_w)
                if plan is None or tokens < plan[0]:
                    plan = (tokens, n_tiles, bounds, chunk_length, overlap)
                if tokens <= token_budget:
                    break
            if plan[0] <= token_budget:
                break
        tokens, n_tiles, bounds, chunk_length, overlap = plan

        print(f"[MMH3 Auto Split Planner] {W * 16}x{H * 16}, {frames_for_tokens(tv)} frames, "
              f"tokens {tokens}/{token_budget}")
        if n_tiles == "1":
            print("  spatial : 1 tile (bypass)")
        else:
            th, tw, ov = _solve_tiles(H, W, n_tiles, tile_overlap_frac)
            rows, cols = len(_tile_starts(H, th, ov)), len(_tile_starts(W, tw, ov))
            print(f"  spatial : {n_tiles} tiles ({rows} row{'s' if rows > 1 else ''} x {cols} col{'s' if cols > 1 else ''}), "
                  f"tile {min(tw, W) * 16}x{min(th, H) * 16} px, overlap {ov * 16} px")
        if len(bounds) == 1:
            print("  temporal: 1 chunk (bypass)")
        else:
            print(f"  temporal: {len(bounds)} chunks, {chunk_length} frames/chunk, overlap {overlap} frames")
        if tokens > token_budget:
            raise ValueError(f"MMH3 Auto Split Planner: no allowed split fits token_budget "
                             f"(smallest needs {tokens} > {token_budget} tokens).")

        if n_tiles != "1":
            model = model.clone()
            model.set_model_unet_function_wrapper(_AutoTiledDiffusionImpl(method, n_tiles, tile_overlap_frac))
        temporal = None
        if len(bounds) > 1:
            temporal = {"chunk_length": chunk_length, "temporal_overlap": overlap, "anchor_strength": anchor_strength}
        return io.NodeOutput(model, temporal)
