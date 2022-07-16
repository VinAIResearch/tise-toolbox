# Gen images for IS and FID 
CUDA_VISIBLE_DEVICES=0 \
python cub_gen_image_from_caption.py \
--caption_input_file "../../image_realism/captions/cub_image_realism_captions.pkl" \
--saved_dir "../../image_realism/images/cub/attngan++" \
--batch_size 32

# Gen images for RP
CUDA_VISIBLE_DEVICES=0 \
python cub_gen_image_from_caption.py \
--caption_input_file "../../text_relevance/captions/CUB_RP_captions.pkl" \
--saved_dir "../../text_relevance/images/cub/attngan++" \
--batch_size 32
