"""Count-based variants of MMH3 Tiled Diffusion and MMH3 Temporal Split Params.

Instead of a tile / chunk size, the user picks how many pieces to cut and the
overlap as a fraction of a piece; sizes are solved from the actual latent.
"""

from comfy_api.latest import io

from .nodes import H3_TEMPORAL_PARAM, compute_segments, frames_for_tokens
from .tiled_diffusion import _H3TiledDiffusionImpl, _ceil2

# n_tiles -> (tiles along the long edge, tiles along the short edge)
TILE_GRIDS = {"1": (1, 1), "2": (2, 1), "3": (3, 1), "4": (2, 2), "6": (3, 2)}


def _tile_size(total, n, overlap):
    """Smallest even tile size so `n` tiles overlapping by `overlap` cover `total`."""
    return _ceil2(-(-(total + (n - 1) * overlap) // n))


class _AutoTiledDiffusionImpl(_H3TiledDiffusionImpl):
    """Solves tile_h / tile_w / overlap from the latent size, then builds the
    tiles exactly like the base implementation."""

    def __init__(self, method, n_tiles, overlap_frac):
        super().__init__(method, 0, 0, 0)
        self.grid = TILE_GRIDS[n_tiles]
        self.overlap_frac = overlap_frac

    def _build(self, payload, video, audio, c):
        H, W = video.shape[3], video.shape[4]
        long_n, short_n = self.grid
        rows, cols = (short_n, long_n) if W >= H else (long_n, short_n)
        # one overlap shared by both axes (base _build contract): grow it until
        # every split axis overlaps by at least overlap_frac of its tile
        ov = 0
        while True:
            th, tw = _tile_size(H, rows, ov), _tile_size(W, cols, ov)
            if all(ov >= self.overlap_frac * t for t, n in ((th, rows), (tw, cols)) if n > 1):
                break
            ov += 2
        self.tile_h, self.tile_w, self.overlap = th, tw, ov
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
                break
            overlap += 17
        print(f"[MMH3 Temporal Split Params Auto] {len(bounds)} chunks over {frame_count} frames "
              f"(chunk_length={chunk_length}, temporal_overlap={overlap})")
        return io.NodeOutput({
            "chunk_length": chunk_length,
            "temporal_overlap": overlap,
            "anchor_strength": anchor_strength,
        })
