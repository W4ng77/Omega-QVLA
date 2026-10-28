"""Common DiT CUDA-Graph adapter, explicitly verified before deployment."""

import torch
from transformers.feature_extraction_utils import BatchFeature


class HeadGraphs:
    def __init__(self, head):
        self.head = head
        self.original = head.get_action
        self.cache = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.replays = 0

    def __call__(self, backbone, action):
        trees = (dict(backbone), dict(action))
        signature = tuple(
            (i, key, tuple(value.shape), str(value.dtype), str(value.device))
            for i, tree in enumerate(trees)
            for key, value in sorted(tree.items())
            if isinstance(value, torch.Tensor)
        )
        for tree in trees:
            for value in tree.values():
                if isinstance(value, torch.Tensor) and not value.is_cuda:
                    raise RuntimeError("DiT graph requires GPU-resident staged inputs")
                if not isinstance(value, (torch.Tensor, int, float, bool, type(None))):
                    raise RuntimeError(f"Unsupported graph input {type(value)}")
        # Non-tensor inputs are captured constants. Reusing a graph after one
        # changes would silently use the first value. Tensor-only signatures
        # remain exactly the same as the frozen experiment implementation.
        constants = tuple(
            (i, key, type(value).__name__, value.hex() if isinstance(value, float) else value)
            for i, tree in enumerate(trees)
            for key, value in sorted(tree.items())
            if not isinstance(value, torch.Tensor)
        )
        if constants:
            signature += (("constants", constants),)
        if signature not in self.cache:
            static = tuple(
                {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in tree.items()}
                for tree in trees
            )
            features = static[0]["backbone_features"]
            noise = torch.zeros(
                (features.shape[0], self.head.config.action_horizon, self.head.config.action_dim),
                device=features.device,
                dtype=getattr(self.head, "_holoq_noise_dtype", features.dtype),
            )

            def invoke():
                # process_backbone_output mutates its BatchFeature dictionary.
                # Fresh shallow wrappers preserve the original static inputs.
                self.head._holoq_external_noise = noise
                try:
                    return self.original(BatchFeature(data=dict(static[0])), BatchFeature(data=dict(static[1])))
                finally:
                    del self.head._holoq_external_noise

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    invoke()
            side.synchronize()
            actual_noise_dtype = getattr(self.head, "_holoq_noise_dtype", noise.dtype)
            if actual_noise_dtype != noise.dtype:
                # First-ever Graph call may precede any eager invocation.
                # Warmup reveals the original post-conditioning noise dtype.
                noise = torch.zeros(noise.shape, device=noise.device, dtype=actual_noise_dtype)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    invoke()
                side.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side, pool=self.pool):
                output = invoke()
            torch.cuda.current_stream().wait_stream(side)
            self.cache[signature] = (static, noise, graph, output)
        static, noise, graph, output = self.cache[signature]
        for src, dst in zip(trees, static):
            for key, value in src.items():
                if isinstance(value, torch.Tensor):
                    dst[key].copy_(value)
        torch.randn(noise.shape, dtype=noise.dtype, device=noise.device, out=noise)
        graph.replay()
        self.replays += 1
        return output


def install(policy):
    head = policy.model.action_head
    graphs = HeadGraphs(head)
    head.get_action = graphs
    return graphs
