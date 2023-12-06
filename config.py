import os

class Config:
    def __init__(self):
        
        self.image_dir = './data/image'

        self.data_dir = './data/annotation'
        self.weight_dir = './weight'

        self.best_crop_data = os.path.join(self.data_dir, 'best_crop')
        self.unlabeled_data = os.path.join(self.data_dir, 'unlabeled_vapnet')
        self.labeled_data = os.path.join(self.data_dir, 'labeled_vapnet')

        self.adjustment_count = 8
        self.fpr_limit = 0.3

        self.gpu_id = 0
        self.num_workers = 4

        self.adjustment_type_list = ['horizontal_shift', 'vertical_shift', 'zoom', 'rotate', 'best']

        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 5e-4

        self.max_epoch = 10000

        self.image_size = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]