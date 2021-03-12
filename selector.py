import queue
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from option import args


class PatchItem:
    def __init__(self, lr: np.ndarray, hr: np.ndarray, distance: int):
        self.lr = lr
        self.hr = hr
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance


class Selector:
    def __init__(self):
        self.sample_rate = 1
        self.patch_size = 64

        self.candidates = list(Path(args.candidates).iterdir())
        self.candidate_pts_range = 3
        self.reference_patch_length = 8
        self.candidate_patch_length = 128

    def _select_reference_patch(self, reference_frame: List[np.ndarray]) -> List[np.ndarray]:
        patch_size = self.patch_size
        current_frame = reference_frame.pop()
        height, width, _ = current_frame.shape

        m, n = width // patch_size, height // patch_size
        mse_table = np.zeros(m * n)
        for i in range(m):
            for j in range(n):
                current_roi = current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                for frame in reference_frame:
                    reference_roi = frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                    mse = np.mean((current_roi - reference_roi) ** 2)
                    mse_table[i * n + j] += mse

        reference_patches = []
        indexes = mse_table.argsort()[::-1][:self.reference_patch_length]
        for idx in indexes:
            i, j = idx // n, idx % n
            reference_patches.append(
                current_frame[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size])
        return reference_patches

    def select_patches(self, reference_frame: List[np.ndarray], pts: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        candidates = []
        patch_size = self.patch_size
        scale = 2
        reference_patches = self._select_reference_patch(reference_frame)
        reference_patches = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in reference_patches]
        # cv2.imshow('reference', reference_patch)

        for candidate in self.candidates:
            for i in range(max(0, pts - self.candidate_pts_range), pts):
                filename = candidate / f'{str(i).zfill(4)}.png'
                hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
                lr = cv2.resize(hr, (960, 540), interpolation=cv2.INTER_CUBIC)
                candidates.append((lr, hr))

        patch_queue = queue.PriorityQueue()
        for ref_patch in reference_patches:
            reference_hash = cv2.img_hash.pHash(ref_patch)[0]
            for lr, hr in candidates:
                height, width, _ = lr.shape
                m, n = width // patch_size, height // patch_size
                for i in range(m):
                    for j in range(n):
                        roi = lr[j * patch_size: (j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
                        hr_roi = hr[j * patch_size * scale:(j + 1) * patch_size * scale,
                                 i * patch_size * scale:(i + 1) * patch_size * scale]
                        roi_hash = cv2.img_hash.pHash(roi)[0]
                        distance = sum([bin(x ^ y).count('1') for x, y in zip(reference_hash, roi_hash)])
                        patch_queue.put(PatchItem(roi, hr_roi, distance))

        patches = []
        for i in range(self.candidate_patch_length):
            item = patch_queue.get()
            patches.append((item.lr, item.hr))
        return patches


class RandomSelector:
    def __init__(self):
        self.candidates = list(Path(args.candidates).iterdir())
        self.patch_size = 64
        self.scale = 2

    def select_patch(self) -> (np.ndarray, np.ndarray):
        candidate = self.candidates[random.randrange(0, len(self.candidates))]
        candidate_size = len(list(candidate.iterdir()))
        frame_id = random.randrange(0, candidate_size)
        frame = cv2.imread(str(candidate / f'{str(frame_id).zfill(4)}.png'))
        scale = self.scale
        patch_size = self.patch_size

        height, width, _ = frame.shape
        height, width = height // scale, width // scale
        lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        x = random.randrange(0, width - patch_size + 1)
        y = random.randrange(0, height - patch_size + 1)
        lr_patch = lr[y:y + patch_size, x:x + patch_size]
        hr_patch = frame[y * scale:(y + patch_size) * scale, x * scale:(x + patch_size) * scale]

        return lr_patch, hr_patch

    def select_patches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        for _ in range(1):
            candidate = self.candidates[random.randrange(0, len(self.candidates))]
            candidate_size = len(list(candidate.iterdir()))
            frame_id = random.randrange(0, candidate_size)
            frame = cv2.imread(str(candidate / f'{str(frame_id).zfill(4)}.png'))
            scale = self.scale
            patch_size = self.patch_size

            height, width, _ = frame.shape
            height, width = height // scale, width // scale
            lr = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            for _ in range(64):
                x = random.randrange(0, width - patch_size + 1)
                y = random.randrange(0, height - patch_size + 1)
                lr_patch = lr[y:y + patch_size, x:x + patch_size]
                hr_patch = frame[y * scale:(y + patch_size) * scale, x * scale:(x + patch_size) * scale]
                patches.append((lr_patch, hr_patch))

        return patches
