import asyncio
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.multiprocessing as mp
from aiortc import RTCPeerConnection, RTCSessionDescription
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import OnlineDataset
from option import args
from process import load_model, train_one_epoch, inference
from selector import Selector, RandomSelector


async def run(frame_queue: mp.Queue, patch_queue: mp.Queue):
    pc = RTCPeerConnection()

    @pc.on('track')
    async def on_track(track):
        frame_id = 0
        while True:
            frame = await track.recv()
            frame = frame.to_ndarray(format='rgb24')
            if frame_id % args.sample_interval == 0:
                frame_queue.put(frame)
                patch_queue.put(frame)
            frame_id += 1

    @pc.on('icegatheringstatechange')
    async def on_icegatheringstatechange():
        print('icegatheringstate: ', pc.iceGatheringState)

    # negotiate
    pc.addTransceiver('video', direction='recvonly')

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # todo: icegatheringstatechange
    r = requests.post(f'http://127.0.0.1:{args.port}/offer', json={
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type,
    })
    answer = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))

    await asyncio.sleep(3600)


def render(frame_queue: mp.Queue, state_queue: mp.Queue):
    torch.cuda.set_device(0)

    pretrain_model = load_model()
    model = load_model()
    model.cuda()
    model.eval()

    result_file = Path(f'result_{int(time.time())}.txt').open('w', encoding='utf8')

    cap = cv2.VideoCapture()
    cap.open(args.hr_video)

    async def show_frame():
        pts = 0
        while True:
            # update model
            buffer = None
            while state_queue.qsize() > 0:
                buffer = state_queue.get()
            if buffer:
                model.load_state_dict(torch.load(buffer))

            # train model
            if not frame_queue.empty():
                frame = frame_queue.get()

                ret, hr_frame = cap.read()
                hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
                for _ in range(args.sample_interval - 1):
                    _, _ = cap.read()

                bicubic_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                pretrain_frame = inference(pretrain_model, frame, 2)
                sr_frame = inference(model, frame, 2)

                bicubic_frame_psnr = psnr(hr_frame, bicubic_frame)
                pretrain_frame_psnr = psnr(hr_frame, pretrain_frame)
                sr_frame_psnr = psnr(hr_frame, sr_frame)

                bicubic_frame_ssim = ssim(hr_frame, bicubic_frame, multichannel=True)
                pretrain_frame_ssim = ssim(hr_frame, pretrain_frame, multichannel=True)
                sr_frame_ssim = ssim(hr_frame, sr_frame, multichannel=True)

                psnr_list = np.array([sr_frame_psnr, pretrain_frame_psnr, bicubic_frame_psnr])
                max_idx = psnr_list.argmax()
                print(f'[pts: {pts}, metric: psnr] ' +
                      f'{"*" if max_idx == 0 else ""}online: {sr_frame_psnr}, ' +
                      f'{"*" if max_idx == 1 else ""}pretrain: {pretrain_frame_psnr}, ' +
                      f'{"*" if max_idx == 2 else ""}bicubic: {bicubic_frame_psnr}')

                ssim_list = np.array([sr_frame_ssim, pretrain_frame_ssim, bicubic_frame_ssim])
                max_idx = ssim_list.argmax()
                print(f'[pts: {pts}, metric: ssim] ' +
                      f'{"*" if max_idx == 0 else ""}online: {sr_frame_ssim}, ' +
                      f'{"*" if max_idx == 1 else ""}pretrain: {pretrain_frame_ssim}, ' +
                      f'{"*" if max_idx == 2 else ""}bicubic: {bicubic_frame_ssim}')

                result = {
                    'pts': pts,
                    'psnr': {
                        'bicubic': bicubic_frame_psnr,
                        'pretrain': pretrain_frame_psnr,
                        'online': sr_frame_psnr,
                    },
                    'ssim': {
                        'bicubic': bicubic_frame_ssim,
                        'pretrain': pretrain_frame_ssim,
                        'online': sr_frame_ssim,
                    }
                }
                result_file.write(json.dumps(result) + ',\n')
                result_file.flush()

                pts += 1

            await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    loop.create_task(show_frame())
    loop.run_forever()


def learn(state_queue: mp.Queue, patch_queue: mp.Queue):
    torch.cuda.set_device(0)
    model = load_model()
    model.cuda()

    online_dataset = OnlineDataset(max_size=2048)

    r_selector = RandomSelector()
    for i in range(args.batch_size * 10):
        online_dataset.put(*r_selector.select_patch())

    selector = Selector()
    loader = DataLoader(dataset=online_dataset, num_workers=1, persistent_workers=True,
                        batch_size=args.batch_size, pin_memory=False, shuffle=True)

    async def update_dataset():
        count = 0
        while True:
            reference_frame = []
            if patch_queue.qsize() >= 2:
                while patch_queue.qsize() > 0:
                    reference_frame.append(patch_queue.get())
                    count += 1

                patches = selector.select_patches(reference_frame, count)
                for lr, hr in patches:
                    online_dataset.put(lr, hr)

            await asyncio.sleep(0)

    async def train():
        while True:
            for _ in range(3):
                train_one_epoch(model, loader, 2)

            # send state dict
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            state_queue.put(buffer)

            await asyncio.sleep(5)

    loop = asyncio.get_event_loop()
    loop.create_task(update_dataset())
    loop.create_task(train())
    loop.run_forever()


def main():
    frame_queue = mp.Queue()
    patch_queue = mp.Queue()
    state_queue = mp.Queue()

    render_process = mp.Process(target=render, args=(frame_queue, state_queue))
    render_process.start()

    train_process = mp.Process(target=learn, args=(state_queue, patch_queue))
    train_process.start()

    asyncio.run(run(frame_queue, patch_queue))


if __name__ == '__main__':
    main()
