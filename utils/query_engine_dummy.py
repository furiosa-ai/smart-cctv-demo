
import random
from ai_util.vision.image_gallery import QueryResult
from utils.query_engine_base import QueryEngineBase


class QueryEngineDummy(QueryEngineBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process_gallery_data(self):
        pass

    def create_query_from_image(self, img):
        return None

    def query(self, query):
        # create random results
        ind = [random.randint(0, self.gallery_data.get_frame_count() - 1) for _ in range(self.topk)]
        dists = sorted([random.random() for _ in range(self.topk)])

        return [QueryResult(idx, idx, dist) for idx, dist in zip(ind, dists)]
