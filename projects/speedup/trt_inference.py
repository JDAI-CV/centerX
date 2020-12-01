import os
os.environ['LD_LIBRARY_PATH'] = "/workspace/TensorRT-7.0.0.11-cuda-10.0/lib"

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import random

from tqdm import tqdm
import time

TRT_LOGGER = trt.Logger()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


class centerX_trt():
    def __init__(self, model_path):
        with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.w = 640 // 4
        self.h = 384 // 4
        # self.input_name = self.ort_sess.get_inputs()[0].name

    def resize_and_pad(self, images, new_shape=(640, 384), stride=32):
        # images: a list of image,[[H,W,C],...,[H,W,C]]
        shape = np.array([image.shape[:2][::-1] for image in images])  # width, height
        if isinstance(new_shape, int):
            ratio = float(new_shape) / np.max(shape, axis=1, keepdims=True)
        else:
            new_shape = np.array(new_shape)
            ratio = np.min(new_shape / shape, axis=1, keepdims=True)  # ratio  = new / old

        new_unpads = shape * ratio
        new_unpads = new_unpads.astype(np.int)
        imgs = np.zeros(tuple([len(images)]) + tuple(new_shape[::-1]) + (3,))

        for k, image, new_unpad in zip(range(len(images)), images, new_unpads):
            new_unpad = tuple(new_unpad)
            w, h = new_unpad
            img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
            imgs[k, :h, :w, :] = img
        imgs = imgs.astype(np.float32)
        return imgs, ratio

    def preprocess(self, images, new_size=(640, 384), stride=32):
        # images: [ndarray(H,W,C),...] in bgr Mode
        images = [image[:, :, ::-1] for image in images]  # to rgb
        pad_images, ratio = self.resize_and_pad(images, new_size, stride)
        return pad_images, ratio

    def postprocess(self, result, ratios, thresh=0.3):
        clses, regs, whs = result
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        bboxes = []

        for cls, reg, wh, ratio in zip(clses, regs, whs, ratios):
            index = np.where(cls >= thresh)
            ratio = 4 / ratio
            score = np.array(cls[index])
            cat = np.array(index[0])
            ctx, cty = index[-1], index[-2]
            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)
            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = x1 * ratio, y1 * ratio, x2 * ratio, y2 * ratio
            bbox = np.stack((cat, score, x1, y1, x2, y2), axis=1).tolist()
            bbox = sorted(bbox, key=lambda x: x[1], reverse=True)
            bboxes.append(bbox)

        return bboxes

    def do_inference(self, batch_size=1):
        # Transfer data from CPU to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async(batch_size=batch_size,
                                   bindings=self.bindings,
                                   stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host.reshape(batch_size,-1,self.h,self.w) for out in self.outputs[::-1]]

    def inferencen_on_images(self, images, new_size=(640, 384), thresh=0.25):
        # images: [ndarray(H,W,C),...] in bgr Mode
        # only support batch = 1, rely on centerX2onnx input , line: 142
        # you can modify your batch in centerX2onnx and generate new model
        assert len(images) == 1

        t0 = time.time()
        pad_images, ratios = self.preprocess(images, new_size)
        pad_images = pad_images.transpose(0, 3, 1, 2)
        self.inputs[0].host = pad_images.reshape(-1)
        t1 = time.time()
        print('preprocess cost', t1 - t0, 's')

        result = self.do_inference(batch_size=1)
        # result = self.ort_sess.run(None, {self.input_name: pad_images})
        t2 = time.time()
        print('forward cost', t2 - t1, 's')
        bboxes = self.postprocess(result, ratios, thresh)
        t3 = time.time()
        print('postprocess cost', t3 - t2, 's')
        return bboxes


if __name__ == '__main__':
    ort_sess = centerX_trt('models/person_det_regnetX400MF_adam_nodeform_KD.engine')
    root = '/export/167/cp/bodytest/low_solution/'
    save_path = 'results/'
    images = os.listdir('/export/167/cp/bodytest/low_solution/')
    for img in tqdm(images):
        img_bgr = cv2.imread(root+img)
        bbox = ort_sess.inferencen_on_images([img_bgr], thresh=0.25)[0]
        for (_,_,x1,y1,x2,y2) in bbox:
           cv2.rectangle(img_bgr,(int(x1),int(y1)),(int(x2),int(y2)),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),2)
        cv2.imwrite(save_path+img, img_bgr)
