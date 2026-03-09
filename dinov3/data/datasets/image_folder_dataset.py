# Simple ImageFolder-style dataset for DINOv3

import os
from typing import Any, Callable, List, Optional, Tuple

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder


class ImageFolderDataset(ExtendedVisionDataset):
    """A minimal dataset for generic image folders.

    Expects a directory structure like:

        root/
          class1/ img1.png, img2.png, ...
          class2/ img3.png, ...

    Class names come from subdirectory names; labels are integer indices.
    """

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self._samples: List[Tuple[str, int]] = []
        self._class_to_idx: dict[str, int] = {}

        if not os.path.isdir(root):
            raise RuntimeError(f"ImageFolderDataset root '{root}' is not a directory")

        # Discover classes from subdirectories
        class_names = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

        for class_idx, class_name in enumerate(class_names):
            self._class_to_idx[class_name] = class_idx
            class_dir = os.path.join(root, class_name)

            for fname in sorted(os.listdir(class_dir)):
                if not self._is_image_file(fname):
                    continue
                path = os.path.join(class_dir, fname)
                self._samples.append((path, class_idx))

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        return filename.lower().endswith(valid_exts)

    def get_image_data(self, index: int) -> bytes:
        path, _ = self._samples[index]
        with open(path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        _, target = self._samples[index]
        return target

    def __len__(self) -> int:
        return len(self._samples)
