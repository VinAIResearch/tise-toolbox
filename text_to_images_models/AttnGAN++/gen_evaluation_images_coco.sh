# Generate images for R-precision
CUDA_VISIBLE_DEVICES=0 \
python coco_gen_image_from_caption.py \
--caption_input_file "../../text_relevance/captions/COCO_RP_captions.pkl" \
--saved_dir "../../text_relevance/images/coco/attngan++" \
--batch_size 16

# Generate images for SOA
CUDA_VISIBLE_DEVICES=0 \
python coco_gen_soa_input_images.py \
--label_file_dir "../../semantic_object_accuracy/captions" \
--saved_dir "../../semantic_object_accuracy/images/attngan++" \
--batch_size 16

# Generate images for IS, FID
CUDA_VISIBLE_DEVICES=0 \
python coco_gen_image_from_caption.py \
--caption_input_file "../../image_realism/captions/coco_image_realism_captions.pkl" \
--saved_dir "../../image_realism/images/coco/attngan++" \
--batch_size 16

# Generate images for PA
CUDA_VISIBLE_DEVICES=0 \
python coco_gen_PA_input_images.py \
--caption_input_file "../../positional_alignment/captions/PA_input_captions.pkl" \
--saved_dir "../../positional_alignment/images/attngan++" \
--batch_size 10

# Generate images for CA
CUDA_VISIBLE_DEVICES=0 \
python coco_gen_image_from_caption.py \
--caption_input_file "../../counting_alignment/captions/CA_input_captions.pkl" \
--saved_dir "../../counting_alignment/images/attngan++" \
--batch_size 16