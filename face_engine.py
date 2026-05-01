# face_engine.py
import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, det_size=(640, 640), providers=None):
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_and_embed(self, bgr_image):
        # returns list of dict {bbox, kps, normed_embedding}
        faces = self.app.get(bgr_image)
        results = []
        for f in faces:
            if f.embedding is None:
                # some models expose .normed_embedding, ensure normalization
                emb = f.normed_embedding if hasattr(f, "normed_embedding") else f.embedding
            else:
                emb = f.embedding
            # normalize to unit vector for cosine similarity
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            results.append({
                "bbox": f.bbox.astype(int).tolist(),
                "kps": f.kps.tolist(),
                "embedding": emb.astype(np.float32)
            })
        return results

    @staticmethod
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
