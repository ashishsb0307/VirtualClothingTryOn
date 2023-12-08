class Options:
    def __init__(self):
        self.load_height = 1024
        self.load_width = 768
        self.semantic_nc = 13
        self.dataset_dir = r'datasets/'
        self.dataset_mode = r'test'
        self.dataset_list = r'test_pairs.txt'
        self.batch_size = 1
        self.workers = 1
        self.semantic_nc = 13
        self.init_type = 'xavier'
        self.init_variance = 0.
        self.checkpoint_dir = r'./checkpoints/'
        self.seg_checkpoint = r'seg_final.pth'
        self.gmm_checkpoint = r'gmm_final.pth'
        self.alias_checkpoint = r'alias_final.pth'
        self.grid_size = 5
        self.num_upsampling_layers = r'most'
        self.ngf = 64
        self.norm_G = 'spectralaliasinstance'
        self.save_dir = '.'
        self.name = 'hello'
