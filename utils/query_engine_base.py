


from ai_util.dataset import ImageDataset


class QueryEngineBase:
    def __init__(self, topk=5) -> None:
        self.topk = topk

    def set_gallery_data(self, path):
        self.gallery_data = ImageDataset(path, limit=None, frame_step=1)
        self.process_gallery_data()
        return self.gallery_data

    def process_gallery_data(self):
        raise NotImplementedError

    def create_query_from_image(self, img):
        raise NotImplementedError

    def query(self, query, topk):
        raise NotImplementedError
