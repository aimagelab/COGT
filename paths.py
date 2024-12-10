from torchvision import transforms
from PIL import Image

PATH = {
    'coco_images_negCLIP': 'coco_images_negCLIP_path',
    'coco_training_meta_negCLIP_parser': 'coco_training_meta_negCLIP_parser_path',
    'coco_val_meta_negCLIP': 'coco_val_meta_negCLIP_path',
    'xvlm_weights': 'xvlm_weights_path',
    'config_xvlm': 'config_xvlm_path',
    'config_swin_xvlm': 'config_swin_xvlm_path'
}

TEST_PATH = {
    "visual_genome_relation": {'images': 'visual_genome_relation_image_path',
                               'metadata': 'visual_genome_relation_metadata_path_metadata_path'},
    "fg_ovd": {
        'images': 'fg_ovd_relation_image_path',
        'metadata': 'fg_ovd_metadata_path_metadata_path'},
    "sugar_crepe": {
        'images': 'sugar_crepe_image_path',
        'metadata': 'sugar_crepe_metadata_path'},
    "colorswap": {
        'images': 'colorswap_image_path',
        'metadata': 'colorswap_metadata_path'
    },
    "visual_genome_attribution": {
        'images': 'visual_genome_attribution_image_path',
        'metadata': 'visual_genome_attribution_metadata_path'},
    "vl_checklist": {
        'images': 'vl_checklist_image_path',
        'metadata': 'vl_checklist_metadata_path'}
}


def list_of_strings(arg):
    return arg.split(',')


def get_xvlm_transform(config):
    return transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
